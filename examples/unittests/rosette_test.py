#! /usr/bin/env python3
import logging
import unittest

from examples.Rosette import Rosette


class TestRosette(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.WARN, format='%(asctime)s %(levelname)-7s %(message)s')
        self.rosette = Rosette(url='http://localhost:8181/rest/v1/')

    def test_ping(self):
        result = self.rosette.ping()
        self.assertIsNotNone(result)
        self.assertEqual(result['message'], 'Rosette API at your service')

    def test_info(self):
        result = self.rosette.info()
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'Rosette API')

    def test_text_embedding(self):
        embedding = self.rosette.text_embedding('Grandma got run over by a reindeer')
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 300)
        self.assertAlmostEqual(embedding[0], 0.041, 3)


if __name__ == '__main__':
    unittest.main()
