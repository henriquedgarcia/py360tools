import pickle
import unittest
from pathlib import Path

import pandas as pd

from utils.hm import position2displacement

"""
Teste Head Movement functions
"""

__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent


class TestPosition2Trajectory(unittest.TestCase):
    def test_position2trajectory(self):
        user_hm = pd.read_csv(f'{__PATH__}/assets/user_hm.csv', index_col=0)
        displacement = position2displacement(user_hm)
        displacement_test = pickle.loads(Path(f'{__PATH__}/assets/displacement_test.pickle').read_bytes())
        self.assertTrue(displacement_test.equals(displacement))


if __name__ == '__main__':
    unittest.main()
