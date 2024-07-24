import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from py360tools.transform import position2displacement
from py360tools.utils import load_test_data

"""
Teste Head Movement functions
"""

__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}'
__ASSETS__.mkdir(parents=True, exist_ok=True)


class TestPosition2Trajectory(unittest.TestCase):
    def test_position2trajectory(self):
        user_hm_file = Path(f'{__PATH__}/user_hm.csv')
        displacement_file = Path(f'{__ASSETS__}/displacement.pickle')

        user_hm = pd.read_csv(user_hm_file, index_col=0)
        displacement = position2displacement(df_positions=user_hm)

        displacement_test = load_test_data(displacement_file, displacement)

        self.assertTrue(np.array_equal(displacement, displacement_test))


if __name__ == '__main__':
    unittest.main()
