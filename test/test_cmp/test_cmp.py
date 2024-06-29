import pickle
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from models import CMP, Viewport
from helper.draw import draw
from models.cmp import (cmp2nmface, nmface2vuface, vuface2xyz_face, xyz2vuface, vuface2nmface, nmface2cmp_face,
                        ea2cmp_face, cmp2ea_face)
from utils.util import test, show

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / 'assets'


def load_nm_file():
    nm_file = Path(f'{__ASSETS__}/assets/nm.pickle')
    
    try:
        nm_test = pickle.loads(nm_file.read_bytes())
    except FileNotFoundError:
        shape = (200, 300)
        nm_test = np.ndarray(np.mgrid[0:shape[0], 0:shape[1]])
        nm_file.write_bytes(pickle.dumps(nm_test))
    return nm_test


def load_nmface_file(nm_test):
    nmface_file = Path(f'{__ASSETS__}/assets/nmface.pickle')
    try:
        nmface_test = pickle.loads(nmface_file.read_bytes())
    except FileNotFoundError:
        nmface_test = cmp2nmface(nm=nm_test)
        nmface_file.write_bytes(pickle.dumps(nmface_test))
    return nmface_test


def load_vuface_file(nmface_test):
    vuface_file = Path(f'{__ASSETS__}/assets/vuface.pickle')
    try:
        vuface_test = pickle.loads(vuface_file.read_bytes())
    except FileNotFoundError:
        vuface_test = nmface2vuface(nmface=nmface_test)
        vuface_file.write_bytes(pickle.dumps(vuface_test))
    return vuface_test


def load_xyz_file(vuface_test):
    xyz_file = Path(f'{__ASSETS__}/assets/xyz.pickle')
    try:
        xyz_face_test = pickle.loads(xyz_file.read_bytes())
    except FileNotFoundError:
        xyz_face_test = vuface2xyz_face(vuface=vuface_test)
        xyz_file.write_bytes(pickle.dumps(xyz_face_test))
    return xyz_face_test


def load_ea_file(nm_test):
    ea_file = Path(f'{__ASSETS__}/assets/ae.pickle')
    try:
        ea_test = pickle.loads(ea_file.read_bytes())
    except FileNotFoundError:
        ea_test, face1 = cmp2ea_face(nm=nm_test)
        ea_file.write_bytes(pickle.dumps(ea_test))
    return ea_test


def load_ea_cmp_file(nm_test):
    ea_cmp_file = Path(f'{__ASSETS__}/assets/ea_cmp.pickle')
    try:
        ea_cmp_face_test = pickle.loads(ea_cmp_file.read_bytes())
    except FileNotFoundError:
        ea_cmp_face_test = cmp2ea_face(nm=nm_test)
        ea_cmp_file.write_bytes(pickle.dumps(ea_cmp_face_test))
    return ea_cmp_face_test


class TestCmp(unittest.TestCase):
    projection: CMP
    frame_array: np.ndarray

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 384, 576

        cls.projection = CMP(tiling='6x4', proj_res=f'{width}x{height}', 
                             vp_shape=np.array([500, 600]), fov=np.array([110, 90]))
        cls.projection.yaw_pitch_roll = np.deg2rad((70, 45, -15))

        # Open Image
        frame_img: Image = Image.open('images/cmp1.png')
        frame_img = frame_img.resize((width, height))
        cls.frame_array = np.array(frame_img)
        load_nm_file()
        load_nmface_file()
        load_vuface_file()
        load_xyz_file()
        load_ea_file()
        load_ea_cmp_file(nm_test)

    nmface_test: np.ndarray
    vuface_test: np.ndarray
    xyz_face_test: tuple[np.ndarray, np.ndarray]
    ea_test: np.ndarray
    ae2cmp_test: np.ndarray
    ea_cmp_face_test: tuple[np.ndarray, np.ndarray]
    cmp2ea_test: np.ndarray

    def test_cmp2mn_face(self):
        nm_test = load_nm_file()
        nmface_test = load_nmface_file(nm_test)
        
        nmface = cmp2nmface(nm=nm_test)
        nm, face = nmface2cmp_face(nmface=nmface)
        nm, face = nmface2cmp_face(nmface=nmface_test)

        self.assertTrue(np.array_equal(nm_test, nm), 'Error in reversion')
        self.assertTrue(np.array_equal(nmface, nmface_test), 'Error in nmface2cmp_face()')
        self.assertTrue(np.array_equal(nm_test, nm), 'Error in cmp2nmface()')

    def teste_nmface2vuface(self):
        vuface = nmface2vuface(nmface=self.nmface_test)
        nmface = vuface2nmface(vuface=vuface)

        msg = ''
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in reversion'
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in nmface2vuface()'

        nmface = vuface2nmface(vuface=self.vuface_test)
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in vuface2nmface()'

        assert msg == '', msg

    def teste_vuface2xyz(self):
        xyz, face = vuface2xyz_face(vuface=self.vuface_test)
        vuface = xyz2vuface(xyz=xyz)

        msg = ''
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in reversion'
        if not np.array_equal(xyz, self.xyz_face_test[0]):
            msg += 'Error in vuface2xyz_face()'

        vuface = xyz2vuface(xyz=self.xyz_face_test[0])
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in xyz2vuface()'

        assert msg == '', msg

    def teste_cmp2ea(self):
        ea, face1 = cmp2ea_face(nm=self.nm_test)
        nm, face2 = ea2cmp_face(ea=ea)

        msg = ''
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in reversion'

        nm, face = ea2cmp_face(ea=self.ea_test)
        if not np.array_equal(ea, self.ea_test):
            msg += 'Error in cmp2ea_face()'
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in ea2cmp_face()'

        assert msg == '', msg
