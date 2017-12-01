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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# parse commandline arguments
op = OptionParser()
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


# print(__doc__)


# op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

logging_level = getattr(logging, opts.logging.upper())
logging.basicConfig(level=logging_level,
                    format='%(asctime)s %(levelname)s %(message)s')


# #############################################################################
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
# Uncomment the following to do the analysis on all the categories
categories = None
logging.info("Loading 20 newsgroups dataset for categories: %s", categories or 'All')

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             remove=('headers', 'footers', 'quotes'),
                             shuffle=True)

logging.info("%d documents, %d categories", len(dataset.data), len(dataset.target_names))

labels = dataset.target
true_k = np.unique(labels).shape[0]

logging.debug("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)

logging.debug("done in %fs" % (time() - t0))
logging.debug("n_samples: %d, n_features: %d" % X.shape)

if opts.n_components:
    logging.debug("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    logging.debug("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    logging.debug("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


# #############################################################################
# Do the actual clustering

best_clusterer = None
best_n_clusters = 0
best_median_score = -1

for n_clusters in range(opts.min_clusters, opts.max_clusters + 1):
    if n_clusters > best_n_clusters * 2 and n_clusters > best_n_clusters + 10:
        break
    silhouette_scores = []
    local_best_clusterer = None
    local_best_score = -1

    for i in range(opts.n_trials):
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=3,
                                    batch_size=1000, max_iter=opts.n_iters,
                                    verbose=opts.verbose)

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

    if median > best_median_score:
        best_median_score = median
        best_n_clusters = n_clusters
        best_clusterer = local_best_clusterer

logging.warning("best_n_clusters: %d  best_silhouette_score: %0.3f" %
      (best_n_clusters, best_median_score))

if best_clusterer and not opts.use_hashing:
    logging.warning("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(best_clusterer.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = best_clusterer.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(best_n_clusters):
        logging.warning("Cluster %d: %s", i,
                     [terms[ind] for ind in order_centroids[i, :10]])
