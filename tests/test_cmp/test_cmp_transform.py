import unittest
from pathlib import Path

import numpy as np

from py360tools.transform import (nm2nmface, nmface2vuface, vuface2xyz_face, xyz2vuface, vuface2nmface,
                                  nmface2nm_face, nm2ea_face)
from py360tools.utils import create_nm_coords, create_test_default

__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}'

__ASSETS__.mkdir(parents=True, exist_ok=True)


class TestCmpTransform(unittest.TestCase):
    nm_test: np.ndarray
    nmface_test: np.ndarray
    vuface_test: np.ndarray
    xyz_face_test: np.ndarray
    ea_face_test: np.ndarray

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 384, 576

        # Load expected values
        nm_file = Path(f'{__ASSETS__}/nm.pickle')
        nmface_file = Path(f'{__ASSETS__}/nmface.pickle')
        vuface_file = Path(f'{__ASSETS__}/vuface.pickle')
        xyz_file = Path(f'{__ASSETS__}/xyz.pickle')
        ea_file = Path(f'{__ASSETS__}/ae.pickle')

        cls.nm_test = create_test_default(nm_file, create_nm_coords, (height, width))
        cls.nmface_test = create_test_default(nmface_file, nm2nmface, nm=cls.nm_test)
        cls.vuface_test = create_test_default(vuface_file, nmface2vuface, nmface=cls.nmface_test)
        cls.xyz_face_test = create_test_default(xyz_file, vuface2xyz_face, vuface=cls.vuface_test)
        cls.ea_face_test = create_test_default(ea_file, nm2ea_face, nm=cls.nm_test)

    def test_cmp2mn_face(self):
        with self.subTest('Testing nm2nmface.'):
            nmface = nm2nmface(nm=self.nm_test)
            self.assertTrue(np.array_equal(self.nmface_test, nmface), 'Error in nm2nmface()')

        with self.subTest('Testing nmface2nm_face.'):
            nm, face = nmface2nm_face(nmface=self.nmface_test)
            self.assertTrue(np.array_equal(self.nm_test, nm), 'Error in nmface2nm_face()')

        with self.subTest('Testing reversion.'):
            nm, face = nmface2nm_face(nmface=nmface)
            self.assertTrue(np.array_equal(self.nm_test, nm), 'Error in reversion')

    def test_nmface2vuface(self):
        with self.subTest('Testing nmface2vuface.'):
            vuface = nmface2vuface(nmface=self.nmface_test)
            self.assertTrue(np.array_equal(self.vuface_test, vuface), 'Error in nmface2vuface()')

        with self.subTest('Testing vuface2nmface.'):
            nmface = vuface2nmface(vuface=self.vuface_test)
            self.assertTrue(np.array_equal(self.nmface_test, nmface), 'Error in vuface2nmface()')

        with self.subTest('Testing vuface2nmface.'):
            nmface = vuface2nmface(vuface=vuface)
            self.assertTrue(np.array_equal(self.nmface_test, nmface), 'Error in reversion')

    def test_vuface2xyz(self):
        with self.subTest('Testing vuface2xyz_face.'):
            xyz, face = vuface2xyz_face(vuface=self.vuface_test)
            self.assertTrue(np.array_equal(self.xyz_face_test[0], xyz), 'Error in vuface2xyz_face()')

        with self.subTest('Testing xyz2vuface.'):
            vuface = xyz2vuface(xyz=self.xyz_face_test[0])
            self.assertTrue(np.array_equal(self.vuface_test, vuface), 'Error in xyz2vuface()')

        with self.subTest('Testing reversion.'):
            vuface = xyz2vuface(xyz=xyz)
            self.assertTrue(np.array_equal(self.vuface_test, vuface), 'Error in reversion')
