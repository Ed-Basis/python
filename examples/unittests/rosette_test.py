#! /usr/bin/env python3
import unittest

from examples.rosette import Rosette


class TestRosette(unittest.TestCase):
    def setUp(self):
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
        result = self.rosette.text_embedding('Grandma got run over by a reindeer')
        self.assertIsNotNone(result)
        embedding = result['embedding']
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 300)
        self.assertAlmostEqual(embedding[0], 0.041, 3)


if __name__ == '__main__':
    unittest.main()
