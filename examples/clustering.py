# """
# =======================================
# Clustering text documents using k-means
# =======================================
#
# This is an example showing how the scikit-learn can be used to cluster
# text embeddings.
#
# It can be noted that k-means (and minibatch k-means) are very sensitive to
# feature scaling. This is visible in the Silhouette Coefficient,
# as this measure seems to suffer from the phenomenon called
# "Concentration of Measure" or "Curse of Dimensionality" for high dimensional
# datasets such as text data.
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

from examples.Rosette import Rosette
from examples.Embeddings import Embeddings
from rosette.api import RosetteException


class Trial(object):
    """
    Results of a single clustering trial, or best of a set of trials.
    """
    n_clusters = 0
    clusterer = None
    score = -1.0

    def __init__(self, n_clusters=n_clusters, clusterer=None, score=-1.0):
        self.n_clusters = n_clusters
        self.clusterer = clusterer
        self.score = score

    def compare(self, trial):
        if trial.score > self.score:
            self.n_clusters = trial.n_clusters
            self.clusterer = trial.clusterer
            self.score = trial.score
        return self


class Clusterer(object):
    embeddings = None
    best = None

    def __init__(self, rosette=None, min_clusters=3, max_clusters=20, n_trials=5, n_iters=100,
                 verbose=False):
        self.rosette = rosette
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_trials = n_trials
        self.n_iters = n_iters
        self.verbose = verbose
        self.best = Trial()

    def run(self, data=None, embeddings=None):
        if embeddings is None:
            if data is None:
                raise RosetteException("badData", "Must provide either data or embeddings", None)
            elif self.rosette is None:
                raise RosetteException("badData", "Must provide either embeddings or Rosette object", None)
            else:
                embeddings = np.asarray([self.rosette.text_embedding(d) for d in data])
        self.embeddings = embeddings
        self._run_clusters(embeddings)
        logging.warning("best n_clusters: %d  score: %0.3f" % (self.best.n_clusters, self.best.score))

    def _run_clusters(self, X):
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            if n_clusters > self.best.n_clusters * 2 and n_clusters > self.best.n_clusters + 10:
                break
            local_best = self._cluster_trials(X, n_clusters)
            self.best.compare(local_best)
        logging.debug("self.best.n_clusters: %d  self.best.score: %0.3f" %
                        (self.best.n_clusters, self.best.score))

    def _cluster_trials(self, X, n_clusters):
        scores = []
        local_best = Trial(clusterer=n_clusters)

        for i in range(self.n_trials):
            trial = self._cluster(X, n_clusters)
            scores.append(trial.score)
            local_best.compare(trial)

        # keep the best-performing clusterer, but return the median score to avoid outliers
        local_best.score = np.median(scores)

        logging.warning("n_clusters: %d  score: %0.3f" % (n_clusters, local_best.score))
        return local_best

    def _cluster(self, X, n_clusters):
        clusterer = MiniBatchKMeans(init='k-means++',
                                    n_clusters=n_clusters, n_init=3, batch_size=1000,
                                    max_iter=self.n_iters,  # max_no_improvement=5,
                                    verbose=self.verbose)
        logging.debug("Clustering data with %s" % clusterer)
        t0 = time()
        clusterer.fit(X)
        elapsed = time() - t0
        score = metrics.silhouette_score(X, clusterer.labels_, sample_size=1000)
        # calinski_harabaz_score seems to be of little use for determining number of clusters,
        #    as a lower number of clusters is always higher
        # calinski_score = metrics.calinski_harabaz_score(X, clusterer.labels_)
        logging.info("elapsed: %0.3fs  n_clusters: %d  score: %0.3f"
                     % (elapsed, n_clusters, score))
        return Trial(n_clusters, clusterer, score)


def _load_sample_data():
    # Load some categories from the training set
    categories = [
        # 'alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
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
    # categories = None
    logging.warning("Loading 20 newsgroups dataset for categories: %s", categories or 'All')
    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 remove=('headers', 'footers', 'quotes'),
                                 shuffle=True)
    logging.info("%d documents, %d categories", len(dataset.data), len(dataset.target_names))
    return dataset.data


def _parse_args(argv):
    # parse commandline arguments
    op = OptionParser()
    op.add_option('--url', type=str, default='http://localhost:8181/rest/v1/', help='Rosette API URL.')
    op.add_option('--file', type=str, help='File containing either numpy text embeddings or text data.')
    op.add_option('--output', type=str, help='Output file for calculated text embeddings.')
    op.add_option('--n-iters', type=int, default=100, help='Maximum number of iterations.')
    op.add_option('--min-clusters', type=int, default=3, help='Minimum number of clusters.')
    op.add_option('--max-clusters', type=int, default=100, help='Maximum number of clusters.')
    op.add_option('--n-trials', type=int, default=5, help='Number of times to run each trial.')
    op.add_option('--logging', type=str, default='WARN', help='Logging level (default WARN).')
    op.add_option('--verbose', action='store_true', dest='verbose', default=False,
                  help='Print progress reports inside k-means algorithm.')
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error('this script takes no arguments.')
        sys.exit(1)

    logging.basicConfig(level=getattr(logging, opts.logging.upper()),
                        format='%(asctime)s %(levelname)s %(message)s')
    return opts


if __name__ == '__main__':
    opts = _parse_args(sys.argv[1:])
    data = None
    embeddings = None
    if opts.file:
        (data, embeddings) = Embeddings.load(opts.file)
    else:
        data = _load_sample_data()
    clusterer = Clusterer(rosette=Rosette(url=opts.url),
                          min_clusters=opts.min_clusters,
                          max_clusters=opts.max_clusters,
                          n_trials=opts.n_trials,
                          n_iters=opts.n_iters,
                          verbose=opts.verbose)
    clusterer.run(data=data, embeddings=embeddings)
    if opts.output:
        Embeddings.save(opts.output, clusterer.embeddings)
