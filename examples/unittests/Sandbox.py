import unittest

import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        list=(3,4,1,5,6)
        n = np.median(list)
        self.assertEqual(4, n)


if __name__ == '__main__':
    unittest.main()
