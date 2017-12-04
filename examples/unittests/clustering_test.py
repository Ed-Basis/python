#! /usr/bin/env python3
import logging
import unittest

import numpy as np

from examples.Rosette import Rosette
from examples.clustering import Clusterer, Trial

# TODO: imports lead to a bunch of ImportWarning's, unclear why


class TestClusterer(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.WARN, format='%(asctime)s %(levelname)-7s %(message)s')

    def test_trial(self):
        a = Trial()
        b = Trial(n_clusters=3, score=0.2)
        c = Trial(n_clusters=5, score=0.1)
        a.compare(b).compare(c)
        self.assertEqual(3, a.n_clusters)
        self.assertEqual(0.2, a.score)

    def test_clusterer(self):
        rosette = Rosette(url='http://localhost:8181/rest/v1/')
        data = ['cookie', 'cake', 'muffin',
                'apple', 'orange', 'banana', 'pear', 'apricot',
                'NASA', 'star', 'black hole', 'telescope',
                'tennis', 'Andre Agassi', 'Martina Navratilova']
        clusterer = Clusterer(rosette=rosette, min_clusters=2, max_clusters=8)
        clusterer.run(data)
        # np.save('data', clusterer.embeddings)
        self.assertEqual(3, clusterer.best.n_clusters)


if __name__ == '__main__':
    unittest.main()
