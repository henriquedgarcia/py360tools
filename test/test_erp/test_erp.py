import pickle
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from models.erp import ERP, erp2vu, vu2ea, vu2erp
from utils.transform import ea2xyz
from utils.util import test, show

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent


class TestErp(unittest.TestCase):
    projection: ERP

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 288, 576

        cls.projection = ERP(tiling='6x4', proj_res=f'{width}x{height}', fov='110x90')
        cls.projection.yaw_pitch_roll = np.deg2rad((70, 0, 0))

        # Open Image
        frame_img: Image = Image.open('images/erp1.png')
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


class TestErpOld(unittest.TestCase):
    nm_test: np.ndarray
    vu_test: np.ndarray
    xyz_test: np.ndarray
    ea_test: np.ndarray
    nm_file = Path('data_test/ERP_nm.pickle')
    vu_file = Path('data_test/ERP_vu.pickle')

    @classmethod
    def setUpClass(cls):
        cls.load_nm_file()
        cls.load_vu_file()
        cls.load_ea_file()
        cls.load_xyz_file()

    @classmethod
    def load_nm_file(cls):
        if cls.nm_file.exists():
            cls.nm_test = pickle.loads(cls.nm_file.read_bytes())
        else:
            shape = (200, 300)
            cls.nm_test = np.array(np.mgrid[0:shape[0], 0:shape[1]])
            cls.nm_file.write_bytes(pickle.dumps(cls.nm_test))

    @classmethod
    def load_vu_file(cls):
        if cls.vu_file.exists():
            cls.vu_test = pickle.loads(cls.vu_file.read_bytes())
        else:
            cls.vu_test = erp2vu(cls.nm_test)
            cls.vu_file.write_bytes(pickle.dumps(cls.vu_test))

    def load_ea_file(self):
        ea_file = Path('data_test/ERP_ae.pickle')
        if ea_file.exists():
            self.ea_test = pickle.load(ea_file.open('rb'))
        else:
            self.ea_test, face1 = vu2ea(self.vu_test)

            with open(ea_file, 'wb') as f:
                pickle.dump(self.ea_test, f)

    def load_xyz_file(self):
        xyz_file = Path('data_test/ERP_xyz.pickle')
        if xyz_file.exists():
            self.xyz_test = pickle.load(xyz_file.open('rb'))
        else:
            self.xyz_test = ea2xyz(self.ea_test)
            with open(xyz_file, 'wb') as f:
                pickle.dump(self.xyz_test, f)

    def test(self):
        test(self.teste_erp2vu)
        test(self.teste_nmface2vuface)
        test(self.teste_vuface2xyz)
        test(self.teste_cmp2ea)

    def teste_erp2vu(self):
        vu = erp2vu(self.nm_test)
        nm = vu2erp(vu)

        msg = ''
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in reversion'
        if not np.array_equal(vu, self.nmface_test):
            msg += 'Error in nmface2cmp_face()'

        nm, face = nmface2cmp_face(self.nmface_test)
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in cmp2nmface()'

        assert msg == '', msg

    def teste_nmface2vuface(self):
        vuface = nmface2vuface(self.nmface_test)
        nmface = vuface2nmface(vuface)

        msg = ''
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in reversion'
        if not np.array_equal(vuface, self.vu_test):
            msg += 'Error in nmface2vuface()'

        nmface = vuface2nmface(self.vu_test)
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in vuface2nmface()'

        assert msg == '', msg

    def teste_vuface2xyz(self):
        xyz, face = vuface2xyz_face(self.vu_test)
        vuface = xyz2vuface(xyz)

        msg = ''
        if not np.array_equal(vuface, self.vu_test):
            msg += 'Error in reversion'
        if not np.array_equal(xyz, self.xyz_face_test[0]):
            msg += 'Error in vuface2xyz_face()'

        vuface = xyz2vuface(self.xyz_face_test[0])
        if not np.array_equal(vuface, self.vu_test):
            msg += 'Error in xyz2vuface()'

        assert msg == '', msg

    def teste_cmp2ea(self):
        ea, face1 = cmp2ea_face(self.nm_test)
        nm, face2 = ea2cmp_face(ea)

        msg = ''
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in reversion'

        nm, face = ea2cmp_face(self.ea_test)
        if not np.array_equal(ea, self.ea_test):
            msg += 'Error in cmp2ea_face()'
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in ea2cmp_face()'

        assert msg == '', msg
