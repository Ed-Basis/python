#! /usr/bin/env python3
import logging
import unittest

from examples.embeddings import Embeddings
from examples.rosette import Rosette


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-7s %(message)s')
        self.embeddings = Embeddings(Rosette(url='http://localhost:8181/rest/v1/'))

    def test_similarity(self):
        text1 = 'Grandma got run over by a reindeer'
        text2 = 'Grandpa got run over by a tractor'
        text3 = 'Wile E. got run over by a steamroller'
        text4 = 'I would gladly pay you Tuesday for a hamburger today'
        result = self.embeddings.similarity(text1, text2)
        self.assertAlmostEqual(result, 0.714, 3)
        result = self.embeddings.similarity(text1, text3)
        self.assertAlmostEqual(result, 0.389, 3)
        result = self.embeddings.similarity(text1, text4)
        self.assertAlmostEqual(result, 0.338, 3)

    def test_coherence(self):
        text1 = 'Grandma got run over by a reindeer'
        text2 = 'Grandpa got run over by a tractor'
        text3 = 'Mama got run over by a puppy'
        text4 = 'Wile E. got run over by a steamroller'
        text5 = 'I would gladly pay you Tuesday for a hamburger today'
        result = self.embeddings.coherence(text1, text2)
        self.assertAlmostEqual(result, 0.714, 3)
        result = self.embeddings.coherence(text1, text2, text3)
        self.assertAlmostEqual(result, 0.680, 3)
        result = self.embeddings.coherence(text1, text2, text3, text4)
        self.assertAlmostEqual(result, 0.554, 3)
        result = self.embeddings.coherence(text1, text2, text3, text4, text5)
        self.assertAlmostEqual(result, 0.444, 3)

    def test_mean_match(self):
        text1 = 'Grandma got run over by a reindeer'
        text2 = 'Grandpa got run over by a tractor'
        text3 = 'Mama got run over by a puppy'
        text4 = 'Wile E. got run over by a steamroller'
        text5 = 'I would gladly pay you Tuesday for a hamburger today'
        category = self.embeddings.mean(text1, text2, text3)
        result = self.embeddings.match(text1, category)
        self.assertAlmostEqual(result, 0.899, 3)
        result = self.embeddings.match(text2, category)
        self.assertAlmostEqual(result, 0.888, 3)
        result = self.embeddings.match(text3, category)
        self.assertAlmostEqual(result, 0.874, 3)
        result = self.embeddings.match(text4, category)
        self.assertAlmostEqual(result, 0.481, 3)
        result = self.embeddings.match(text5, category)
        self.assertAlmostEqual(result, 0.372, 3)

    def test_categories(self):
        category1 = self.embeddings.mean('cookie', 'cake', 'muffin')
        category2 = self.embeddings.mean('apple', 'orange', 'banana', 'pear')
        category3 = self.embeddings.mean('tennis', 'Andre Agassi')
        self.compare('cake', category1, 1)
        self.compare('cake', category2, 2)
        self.compare('cake', category3, 3)
        self.compare('apple', category1, 1)
        self.compare('apple', category2, 2)
        self.compare('apple', category3, 3)
        self.compare('apricot', category1, 1)
        self.compare('apricot', category2, 2)
        self.compare('apricot', category3, 3)
        self.compare('Martina Navratilova', category1, 1)
        self.compare('Martina Navratilova', category2, 2)
        self.compare('Martina Navratilova', category3, 3)

    def compare(self, term, category, category_index):
        logging.debug("compare %s:%d", term, category_index)
        similarity = self.embeddings.match(term, category, 'eng')
        logging.info("match %s:category%d %f", term, category_index, similarity)


if __name__ == '__main__':
    unittest.main()
