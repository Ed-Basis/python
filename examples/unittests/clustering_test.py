#! /usr/bin/env python3
import logging
import unittest

from examples.clustering import Clusterer
from examples.embeddings import Embeddings
from examples.rosette import Rosette


class TestClusterer(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-7s %(message)s')
        self.clusterer = Clusterer(rosette=Rosette(url='http://localhost:8181/rest/v1/'),
                                   opts=None)

    def test_embeddings(self):
        texts = [
            'Grandma got run over by a reindeer',
            'Grandpa got run over by a tractor',
            'Wile E. got run over by a steamroller',
            'I would gladly pay you Tuesday for a hamburger today']
        embeddings = self.clusterer.embeddings(texts)
        self.assertEqual(4, len(embeddings))
        self.assertEqual(300, len(embeddings[0]))


if __name__ == '__main__':
    unittest.main()
