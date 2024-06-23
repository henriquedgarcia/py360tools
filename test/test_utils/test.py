import unittest

import numpy as np

from models.tiling import Tile


class TestTiling(unittest.TestCase):
    def test_tile(self):
        tile = Tile(0, np.array([4, 6]), np.array([200, 200]))
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
