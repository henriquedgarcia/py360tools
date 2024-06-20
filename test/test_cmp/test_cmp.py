import pickle
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from projections.cmp import (CMP)
from utils.transform import cmp_cmp2nmface, cmp_nmface2vuface, cmp_vuface2xyz_face, cmp_xyz2vuface, cmp_vuface2nmface, \
    cmp_nmface2cmp_face, cmp_ea2cmp_face, cmp_cmp2ea_face
from utils.util import test, show

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent


class TestCmp(unittest.TestCase):
    projection: CMP

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 384, 576

        cls.projection = CMP(tiling='6x4', proj_res=f'{width}x{height}', fov='110x90')
        cls.projection.yaw_pitch_roll = np.deg2rad((70, 0, 0))

        # Open Image
        frame_img: Image = Image.open('images/cmp1.png')
        frame_img = frame_img.resize((width, height))
        cls.frame_array = np.array(frame_img)

    draw_all_tiles_borders_test_file = Path(f'{__PATH__}/assets/draw_all_tiles_borders_test_file.pickle')

    def test_draw_all_tiles_borders(self):
        draw_all_tiles_borders = self.projection.draw_all_tiles_borders()

        try:
            draw_all_tiles_borders_test = pickle.loads(self.draw_all_tiles_borders_test_file.read_bytes())
        except FileNotFoundError:
            self.draw_all_tiles_borders_test_file.write_bytes(pickle.dumps(draw_all_tiles_borders))
            draw_all_tiles_borders_test = pickle.loads(self.draw_all_tiles_borders_test_file.read_bytes())

        self.assertTrue(np.alltrue(draw_all_tiles_borders_test == draw_all_tiles_borders))

    draw_vp_borders_test_file = Path(f'{__PATH__}/assets/draw_vp_borders_test_file.pickle')

    def test_draw_vp_borders(self):
        draw_vp_borders = self.projection.draw_vp_borders()

        try:
            draw_vp_borders_test = pickle.loads(self.draw_vp_borders_test_file.read_bytes())
        except FileNotFoundError:
            self.draw_vp_borders_test_file.write_bytes(pickle.dumps(draw_vp_borders))
            draw_vp_borders_test = pickle.loads(self.draw_vp_borders_test_file.read_bytes())

        self.assertTrue(np.alltrue(draw_vp_borders_test == draw_vp_borders))

    draw_vp_mask_test_file = Path(f'{__PATH__}/assets/draw_vp_mask_test_file.pickle')

    def test_draw_vp_mask(self):
        draw_vp_mask = self.projection.draw_vp_mask()

        try:
            draw_vp_mask_test = pickle.loads(self.draw_vp_mask_test_file.read_bytes())
        except FileNotFoundError:
            self.draw_vp_mask_test_file.write_bytes(pickle.dumps(draw_vp_mask))
            draw_vp_mask_test = pickle.loads(self.draw_vp_mask_test_file.read_bytes())

        self.assertTrue(np.alltrue(draw_vp_mask_test == draw_vp_mask))

    draw_vp_tiles_test_file = Path(f'{__PATH__}/assets/draw_vp_tiles_test_file.pickle')

    def test_draw_vp_tiles(self):
        draw_vp_tiles = self.projection.draw_vp_tiles()

        try:
            draw_vp_tiles_test = pickle.loads(self.draw_vp_tiles_test_file.read_bytes())
        except FileNotFoundError:
            self.draw_vp_tiles_test_file.write_bytes(pickle.dumps(draw_vp_tiles))
            draw_vp_tiles_test = pickle.loads(self.draw_vp_tiles_test_file.read_bytes())

        self.assertTrue(np.alltrue(draw_vp_tiles_test == draw_vp_tiles))

    get_vptiles_test_file = Path(f'{__PATH__}/assets/get_vptiles_test_file.pickle')

    def test_get_vptiles(self):
        get_vptiles = self.projection.get_vptiles()
        get_vptiles = np.array(get_vptiles)

        try:
            get_vptiles_test = pickle.loads(self.get_vptiles_test_file.read_bytes())
        except FileNotFoundError:
            self.get_vptiles_test_file.write_bytes(pickle.dumps(get_vptiles))
            get_vptiles_test = pickle.loads(self.get_vptiles_test_file.read_bytes())

        self.assertTrue(np.alltrue(get_vptiles_test == get_vptiles))

    get_viewport_image_test_file = Path(f'{__PATH__}/assets/get_viewport_image_test_file.pickle')

    def test_get_viewport_image(self):
        get_viewport_image = self.projection.get_viewport_image(self.frame_array)

        try:
            get_viewport_image_test = pickle.loads(self.get_viewport_image_test_file.read_bytes())
        except FileNotFoundError:
            self.get_viewport_image_test_file.write_bytes(pickle.dumps(get_viewport_image))
            get_viewport_image_test = pickle.loads(self.get_viewport_image_test_file.read_bytes())

        self.assertTrue(np.alltrue(get_viewport_image_test == get_viewport_image))


class TestCMP:
    nm_test: np.ndarray
    nmface_test: np.ndarray
    vuface_test: np.ndarray
    xyz_face_test: tuple[np.ndarray, np.ndarray]
    ea_test: np.ndarray
    ae2cmp_test: np.ndarray
    ea_cmp_face_test: tuple[np.ndarray, np.ndarray]
    cmp2ea_test: np.ndarray

    def __init__(self):
        self.load_arrays()
        self.test()

    def load_arrays(self):
        self.load_nm_file()
        self.load_nmface_file()
        self.load_vuface_file()
        self.load_xyz_file()
        self.load_ea_file()
        self.load_ea_cmp_file()

    def test(self):
        test(self.teste_cmp2mn_face)
        test(self.teste_nmface2vuface)
        test(self.teste_vuface2xyz)
        test(self.teste_cmp2ea)

    def teste_cmp2mn_face(self):
        nmface = cmp_cmp2nmface(nm=self.nm_test)
        nm, face = cmp_nmface2cmp_face(nmface=nmface)

        msg = ''
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in reversion'
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in nmface2cmp_face()'

        nm, face = cmp_nmface2cmp_face(nmface=self.nmface_test)
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in cmp2nmface()'

        assert msg == '', msg

    def teste_nmface2vuface(self):
        vuface = cmp_nmface2vuface(nmface=self.nmface_test)
        nmface = cmp_vuface2nmface(vuface=vuface)

        msg = ''
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in reversion'
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in nmface2vuface()'

        nmface = cmp_vuface2nmface(vuface=self.vuface_test)
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in vuface2nmface()'

        assert msg == '', msg

    def teste_vuface2xyz(self):
        xyz, face = cmp_vuface2xyz_face(vuface=self.vuface_test)
        vuface = cmp_xyz2vuface(xyz=xyz)

        msg = ''
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in reversion'
        if not np.array_equal(xyz, self.xyz_face_test[0]):
            msg += 'Error in vuface2xyz_face()'

        vuface = cmp_xyz2vuface(xyz=self.xyz_face_test[0])
        if not np.array_equal(vuface, self.vuface_test):
            msg += 'Error in xyz2vuface()'

        assert msg == '', msg

    def teste_cmp2ea(self):
        ea, face1 = cmp_cmp2ea_face(nm=self.nm_test)
        nm, face2 = cmp_ea2cmp_face(ea=ea)

        msg = ''
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in reversion'

        nm, face = cmp_ea2cmp_face(ea=self.ea_test)
        if not np.array_equal(ea, self.ea_test):
            msg += 'Error in cmp2ea_face()'
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in ea2cmp_face()'

        assert msg == '', msg

    def load_nm_file(self):
        nm_file = Path('data_test/nm.pickle')
        if nm_file.exists():
            self.nm_test = pickle.load(nm_file.open('rb'))
        else:
            shape = (200, 300)
            self.nm_test = np.ndarray(np.mgrid[0:shape[0], 0:shape[1]])
            with open(nm_file, 'wb') as f:
                pickle.dump(self.nm_test, f)

    def load_nmface_file(self):
        nmface_file = Path('data_test/nmface.pickle')
        if nmface_file.exists():
            self.nmface_test = pickle.load(nmface_file.open('rb'))
        else:
            self.nmface_test = cmp_cmp2nmface(nm=self.nm_test)
            with open(nmface_file, 'wb') as f:
                pickle.dump(self.nmface_test, f)

    def load_vuface_file(self):
        vuface_file = Path('data_test/vuface.pickle')
        if vuface_file.exists():
            self.vuface_test = pickle.load(vuface_file.open('rb'))
        else:
            self.vuface_test = cmp_nmface2vuface(nmface=self.nmface_test)
            with open(vuface_file, 'wb') as f:
                pickle.dump(self.vuface_test, f)

    def load_xyz_file(self):
        xyz_file = Path('data_test/xyz.pickle')
        if xyz_file.exists():
            self.xyz_face_test = pickle.load(xyz_file.open('rb'))
        else:
            self.xyz_face_test = cmp_vuface2xyz_face(vuface=self.vuface_test)
            with open(xyz_file, 'wb') as f:
                pickle.dump(self.xyz_face_test, f)

    def load_ea_file(self):
        ea_file = Path('data_test/ae.pickle')
        if ea_file.exists():
            self.ea_test = pickle.load(ea_file.open('rb'))
        else:
            self.ea_test, face1 = cmp_cmp2ea_face(nm=self.nm_test)

            with open(ea_file, 'wb') as f:
                pickle.dump(self.ea_test, f)

    def load_ea_cmp_file(self):
        ea_cmp_file = Path('data_test/ea_cmp.pickle')

        if ea_cmp_file.exists():
            self.ea_cmp_face_test = pickle.load(ea_cmp_file.open('rb'))
        else:
            self.ea_cmp_face_test = cmp_ea2cmp_face(ea=self.ea_test)

            with open(ea_cmp_file, 'wb') as f:
                pickle.dump(self.ea_cmp_face_test, f)
