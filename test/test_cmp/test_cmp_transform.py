import pickle
import unittest
from pathlib import Path

import numpy as np

from models import CMP
from transform.cmp_transform import cmp2nmface, nmface2vuface, vuface2xyz_face, xyz2vuface, vuface2nmface, \
    nmface2cmp_face, \
    ea2cmp_face, cmp2ea_face
from utils.util import show

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / 'assets'

nm_file = Path(f'{__ASSETS__}/TestCmpTransform/nm.pickle')
nmface_file = Path(f'{__ASSETS__}/TestCmpTransform/nmface.pickle')
vuface_file = Path(f'{__ASSETS__}/TestCmpTransform/vuface.pickle')
xyz_file = Path(f'{__ASSETS__}/TestCmpTransform/xyz.pickle')
ea_file = Path(f'{__ASSETS__}/TestCmpTransform/ae.pickle')
ea_cmp_file = Path(f'{__ASSETS__}/TestCmpTransform/ea_cmp.pickle')

# erp '144x72', '288x144','432x216','576x288'
# cmp '144x96', '288x192','432x288','576x384'
height, width = 384, 576

projection = CMP(tiling='6x4', proj_res=f'{width}x{height}',
                 vp_shape=np.array([360, 440]),
                 fov=np.array([np.deg2rad(90), np.deg2rad(110)]))
projection.yaw_pitch_roll = np.deg2rad((70, 45, -15))


class TestCmpTransform(unittest.TestCase):
    nm_test: np.ndarray
    nmface_test: np.ndarray
    vuface_test: np.ndarray
    xyz_face_test: np.ndarray
    ea_test: np.ndarray
    ea_cmp_file: np.ndarray
    projection: CMP

    @classmethod
    def setUpClass(cls):
        # Load expected values
        cls.nm_test = load_nm_file()
        cls.nmface_test = load_nmface_file(cls.nm_test)
        cls.vuface_test = load_vuface_file(cls.nmface_test)
        cls.xyz_face_test = load_xyz_file(cls.vuface_test)
        cls.ea_test = load_ea_file(cls.nm_test)
        cls.ea_cmp_file = load_ea_cmp_file(cls.nm_test)

    def test_cmp2mn_face(self):
        with self.subTest('Testing cmp2nmface.'):
            nmface = cmp2nmface(nm=self.nm_test)
            self.assertTrue(np.array_equal(self.nmface_test, nmface), 'Error in cmp2nmface()')

        with self.subTest('Testing nmface2cmp_face.'):
            nm, face = nmface2cmp_face(nmface=self.nmface_test)
            self.assertTrue(np.array_equal(self.nm_test, nm), 'Error in nmface2cmp_face()')

        with self.subTest('Testing reversion.'):
            nm, face = nmface2cmp_face(nmface=nmface)
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
            self.assertTrue(np.array_equal(self.xyz_face_test[0], xyz), 'Error in nmface2vuface()')

        with self.subTest('Testing xyz2vuface.'):
            vuface = xyz2vuface(xyz=self.xyz_face_test[0])
            self.assertTrue(np.array_equal(self.vuface_test, vuface), 'Error in nmface2vuface()')

        with self.subTest('Testing reversion.'):
            vuface = xyz2vuface(xyz=xyz)
            self.assertTrue(np.array_equal(self.vuface_test, vuface), 'Error in reversion')

    def test_cmp2ea(self):
        with self.subTest('Testing cmp2ea_face.'):
            ea, face1 = cmp2ea_face(nm=self.nm_test)
            self.assertTrue(np.array_equal(self.ea_test, ea), 'Error in cmp2ea_face().')

        with self.subTest('Testing ea2cmp_face.'):
            nm, face = ea2cmp_face(ea=self.ea_test)
            self.assertTrue(np.array_equal(self.nm_test, nm), 'Error in ea2cmp_face().')

        with self.subTest('Testing reversion.'):
            nm, face2 = ea2cmp_face(ea=ea)
            self.assertTrue(np.array_equal(self.nm_test, nm), 'Error in reversion')


def load_nm_file():
    try:
        nm_test = pickle.loads(nm_file.read_bytes())
    except FileNotFoundError:
        shape = (200, 300)
        nm_test = np.mgrid[0:shape[0], 0:shape[1]]
        nm_file.write_bytes(pickle.dumps(nm_test))
    return nm_test


def load_nmface_file(nm_test):
    try:
        nmface_test = pickle.loads(nmface_file.read_bytes())
    except FileNotFoundError:
        nmface_test = cmp2nmface(nm=nm_test)
        nmface_file.write_bytes(pickle.dumps(nmface_test))
    return nmface_test


def load_vuface_file(nmface_test):
    try:
        vuface_test = pickle.loads(vuface_file.read_bytes())
    except FileNotFoundError:
        vuface_test = nmface2vuface(nmface=nmface_test)
        vuface_file.write_bytes(pickle.dumps(vuface_test))
    return vuface_test


def load_xyz_file(vuface_test):
    try:
        xyz_face_test = pickle.loads(xyz_file.read_bytes())
    except FileNotFoundError:
        xyz_face_test = vuface2xyz_face(vuface=vuface_test)
        xyz_file.write_bytes(pickle.dumps(xyz_face_test))
    return xyz_face_test


def load_ea_file(nm_test):
    try:
        ea_test = pickle.loads(ea_file.read_bytes())
    except FileNotFoundError:
        ea_test, face1 = cmp2ea_face(nm=nm_test)
        ea_file.write_bytes(pickle.dumps(ea_test))
    return ea_test


def load_ea_cmp_file(nm_test):
    try:
        ea_cmp_face_test = pickle.loads(ea_cmp_file.read_bytes())
    except FileNotFoundError:
        ea_cmp_face_test = cmp2ea_face(nm=nm_test)
        ea_cmp_file.write_bytes(pickle.dumps(ea_cmp_face_test))
    return ea_cmp_face_test
