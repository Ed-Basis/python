# """
# =======================================
# Clustering text documents using k-means
# =======================================
#
# This is an example showing how the scikit-learn can be used to cluster
# documents by topics using a bag-of-words approach. This example uses
# a scipy.sparse matrix to store the features instead of standard numpy arrays.
#
# Two feature extraction methods can be used in this example:
#
#   - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
#     frequent words to features indices and hence compute a word occurrence
#     frequency (sparse) matrix. The word frequencies are then reweighted using
#     the Inverse Document Frequency (IDF) vector collected feature-wise over
#     the corpus.
#
#   - HashingVectorizer hashes word occurrences to a fixed dimensional space,
#     possibly with collisions. The word count vectors are then normalized to
#     each have l2-norm equal to one (projected to the euclidean unit-ball) which
#     seems to be important for k-means to work in high dimensional space.
#
#     HashingVectorizer does not provide IDF weighting as this is a stateless
#     model (the fit method does nothing). When IDF weighting is needed it can
#     be added by pipelining its output to a TfidfTransformer instance.
#
# Two algorithms are demoed: ordinary k-means and its more scalable cousin
# minibatch k-means.
#
# Additionally, latent semantic analysis can also be used to reduce dimensionality
# and discover latent patterns in the data.
#
# It can be noted that k-means (and minibatch k-means) are very sensitive to
# feature scaling and that in this case the IDF weighting helps improve the
# quality of the clustering by quite a lot as measured against the "ground truth"
# provided by the class label assignments of the 20 newsgroups dataset.
#
# This improvement is not visible in the Silhouette Coefficient which is small
# for both as this measure seem to suffer from the phenomenon called
# "Concentration of Measure" or "Curse of Dimensionality" for high dimensional
# datasets such as text data. Other measures such as V-measure and Adjusted Rand
# Index are information theoretic based evaluation scores: as they are only based
# on cluster assignments rather than distances, hence not affected by the curse
# of dimensionality.
#
# Note: as k-means is optimizing a non-convex objective function, it will likely
# end up in a local optimum. Several runs with independent random init might be
# necessary to get a good convergence.
#
# """

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
#         Ed Kenschaft (Basis Technology)
# License: BSD 3 clause

from __future__ import print_function

import logging
import sys
from optparse import OptionParser
from time import time

import numpy as np
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups

from examples.rosette import Rosette


class Clusterer(object):
    rosette = None
    opts = None

    best_clusterer = None
    best_n_clusters = 0
    best_median_score = -1

    def __init__(self, rosette, opts):
        self.rosette = rosette
        self.opts = opts

    def embeddings(self, data):
        return [self.rosette.text_embedding(d)['embedding'] for d in data]

    def run(self, data):
        text_embeddings = self.embeddings(data)
        for n_clusters in range(self.opts.min_clusters, self.opts.max_clusters + 1):
            if n_clusters > self.best_n_clusters * 2 and n_clusters > self.best_n_clusters + 10:
                break
            silhouette_scores = []
            local_best_clusterer = None
            local_best_score = -1

            for i in range(self.opts.n_trials):
                clusterer = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=3,
                                            batch_size=1000, max_iter=self.opts.n_iters,
                                            verbose=self.opts.verbose)

                logging.debug("Clustering sparse data with %s" % clusterer)
                t0 = time()
                clusterer.fit(X)
                elapsed = time() - t0
                logging.debug("elapsed: %0.3fs  " % (elapsed))

                logging.debug("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, clusterer.labels_))
                logging.debug("Completeness: %0.3f" % metrics.completeness_score(labels, clusterer.labels_))
                logging.debug("V-measure: %0.3f" % metrics.v_measure_score(labels, clusterer.labels_))
                logging.debug("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, clusterer.labels_))
                silhouette_score = metrics.silhouette_score(X, clusterer.labels_, sample_size=1000)
                logging.info("elapsed: %0.3fs  n_clusters: %d  silhouette_score: %0.3f"
                             % (elapsed, n_clusters, silhouette_score))
                silhouette_scores.append(silhouette_score)

                if silhouette_score > local_best_score:
                    local_best_score = silhouette_score
                    local_best_clusterer = clusterer

            median = np.median(silhouette_scores)
            logging.info("n_clusters: %d  median_silhouette_score: %0.3f" % (n_clusters, median))

            if median > self.best_median_score:
                self.best_median_score = median
                self.best_n_clusters = n_clusters
                self.best_clusterer = local_best_clusterer

        logging.warning("self.best_n_clusters: %d  self.best_silhouette_score: %0.3f" %
                        (self.best_n_clusters, self.best_median_score))

        # if self.best_clusterer and not self.opts.use_hashing:
        #     logging.warning("Top terms per cluster:")
        #
        #     if self.opts.n_components:
        #         original_space_centroids = svd.inverse_transform(self.best_clusterer.cluster_centers_)
        #         order_centroids = original_space_centroids.argsort()[:, ::-1]
        #     else:
        #         order_centroids = self.best_clusterer.cluster_centers_.argsort()[:, ::-1]
        #
        #     terms = vectorizer.get_feature_names()
        #     for i in range(self.best_n_clusters):
        #         logging.warning("Cluster %d: %s", i,
        #                         [terms[ind] for ind in order_centroids[i, :10]])


def parse_args(argv):
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--url", type=str, default='http://localhost:8181/rest/v1/',
                  help="Rosette API URL.")
    op.add_option("--lsa",
                  dest="n_components", type="int", default=300,
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--n-iters", type=int, default=100,
                  help="Maximum number of iterations.")
    op.add_option("--min-clusters", type=int, default=3,
                  help="Minimum number of clusters.")
    op.add_option("--max-clusters", type=int, default=40,
                  help="Maximum number of clusters.")
    op.add_option("--n-trials", type=int, default=7,
                  help="Number of times to run each trial.")
    op.add_option("--logging", type=str, default='WARN',
                  help="Logging level (default WARN).")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)

    logging.basicConfig(level=getattr(logging, opts.logging.upper()),
                        format='%(asctime)s %(levelname)s %(message)s')

    return opts


def load_data():
    # Load some categories from the training set
    categories = [
        # 'alt.atheism',
        # 'comp.graphics',
        # 'comp.os.ms-windows.misc',
        # 'comp.sys.ibm.pc.hardware',
        # 'comp.sys.mac.hardware',
        # 'comp.windows.x',
        # 'misc.forsale',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space',
        # 'soc.religion.christian',
        # 'talk.politics.guns',
        # 'talk.politics.mideast',
        # 'talk.politics.misc',
        # 'talk.religion.misc',
    ]
    # Uncomment the following to include all categories
    categories = None
    logging.info("Loading 20 newsgroups dataset for categories: %s", categories or 'All')

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 remove=('headers', 'footers', 'quotes'),
                                 shuffle=True)

    logging.info("%d documents, %d categories", len(dataset.data), len(dataset.target_names))


if __name__ == '__main__':
    opts = parse_args(sys.argv[1:])
    rosette = Rosette(url=opts.url)
    dataset = load_data()
    clusterer = Clusterer(dataset.data, opts)
