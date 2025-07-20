import unittest
from pathlib import Path

import numpy as np

from py360tools.assets.projection_erp import ERP
from py360tools.transform import ea2xyz, xyz2ea
from py360tools.utils import create_nm_coords, create_test_default

__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}'
__ASSETS__.mkdir(parents=True, exist_ok=True)


class TestErpTransform(unittest.TestCase):
    nm_test: np.ndarray
    vu_test: np.ndarray
    ea_test: np.ndarray
    xyz_test: np.ndarray

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 288, 576

        # Load expected values
        nm_file = Path(f'{__ASSETS__}/nm.pickle')
        vu_file = Path(f'{__ASSETS__}/vu.pickle')
        ea_file = Path(f'{__ASSETS__}/ae.pickle')
        xyz_file = Path(f'{__ASSETS__}/xyz.pickle')

        # Obs: sometimes vu_test and ea_test need to be rounded (8 digits is enough). 8 digits is good precision: One
        # pixel in 4K (4320x2160) resolution has 2*pi*r / 4320 = 0,00145444 rads in the equator with r=1.
        # And in pole the last pixel center has r = 2*pi*1 / 4320 / 2 = 0.000727220521664304. So the pole pixel has
        # 0.00000108 rads.

        cls.nm_test = create_test_default(nm_file, create_nm_coords, shape=(height, width))
        cls.vu_test = create_test_default(vu_file, ERP.nm2vu, nm=cls.nm_test)
        cls.ea_test = create_test_default(ea_file, ERP.vu2ea, vu=cls.vu_test)
        cls.xyz_test = create_test_default(xyz_file, ea2xyz, ea=cls.ea_test)

    def test_nm2vu(self):
        with self.subTest('Testing nm2vu.'):
            vu = ERP.nm2vu(nm=self.nm_test)
            self.assertTrue(np.array_equal(self.vu_test, vu), 'Error in nm2vu()')

        with self.subTest('Testing vu2nm.'):
            nm = ERP.vu2nm(vu=self.vu_test)
            self.assertTrue(np.array_equal(self.nm_test, nm), 'Error in vu2nm()')

        with self.subTest('Testing reversion.'):
            nm = ERP.vu2nm(vu=vu)
            self.assertTrue(np.array_equal(self.nm_test, nm), 'Error in reversion')

    def teste_vu2ea(self):
        with self.subTest('Testing vu2ea.'):
            ea = ERP.vu2ea(vu=self.vu_test)
            self.assertTrue(np.array_equal(self.ea_test, ea), 'Error in vu2ea()')

        with self.subTest('Testing ea2vu.'):
            vu = ERP.ea2vu(ea=self.ea_test)
            self.assertTrue(np.array_equal(self.vu_test.round(8), vu.round(8)), 'Error in ea2vu()')

        with self.subTest('Testing reversion.'):
            vu = ERP.ea2vu(ea=ea)
            self.assertTrue(np.array_equal(self.vu_test.round(8), vu.round(8)), 'Error in ea2vu()')

    def teste_ea2xyz(self):
        with self.subTest('Testing ea2xyz.'):
            xyz = ea2xyz(ea=self.ea_test)
            self.assertTrue(np.array_equal(self.xyz_test, xyz), 'Error in ea2xyz()')

        with self.subTest('Testing xyz2ea.'):
            ea = xyz2ea(xyz=self.xyz_test)
            self.assertTrue(np.array_equal(self.ea_test.round(8), ea.round(8)), 'Error in xyz2ea()')

        with self.subTest('Testing reversion.'):
            ea = xyz2ea(xyz=xyz)
            self.assertTrue(np.array_equal(self.ea_test.round(8), ea.round(8)), 'Error in reversion')
