import pickle
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from py360tools import CMP, Viewport, show
from py360tools.utils import load_test_data

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}'
__ASSETS__.mkdir(parents=True, exist_ok=True)

xyz_file = Path(f'{__ASSETS__}/xyz.pickle')
vp_img_file = Path(f'{__ASSETS__}/vp_img.pickle')
vptiles_file = Path(f'{__ASSETS__}/vptiles.pickle')


class TestCmp(unittest.TestCase):
    projection: CMP
    frame_array: np.ndarray
    vptiles_test_data: list

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 384, 576

        cls.projection = CMP(tiling='6x4', proj_res=f'{width}x{height}')
        cls.viewport = Viewport('440x360', '110x90', cls.projection)
        cls.viewport.yaw_pitch_roll = np.deg2rad((0, 0, 0))

        frame_img: Image = Image.open('images/cmp1.png')
        frame_img = frame_img.resize((width, height))
        cls.frame_array = np.array(frame_img)

        # Load expected values
        cls.xyz_test_data = load_test_data(xyz_file, cls.projection.xyz)
        cls.vp_img_test_data = load_test_data(vp_img_file, cls.viewport.extract_viewport(cls.frame_array))
        cls.vptiles_test_data = load_test_data(vptiles_file, list(map(int, cls.viewport.get_vptiles())))

    def test_nm2xyz(self):
        xyz = self.projection.nm2xyz(self.projection.nm)
        self.assertTrue(np.array_equal(xyz, self.xyz_test_data))

    def test_xyz2nm(self):
        nm = self.projection.xyz2nm(self.xyz_test_data)
        self.assertTrue(np.array_equal(nm, self.projection.nm))

    def test_extract_viewport(self):
        vp_img = self.viewport.extract_viewport(self.frame_array)
        # show(vp_img)
        self.assertTrue(np.array_equal(vp_img, self.vp_img_test_data))

    def test_get_vptiles(self):
        get_vptiles = list(self.viewport.get_vptiles())
        # print(get_vptiles)
        self.assertTrue(np.array_equal(self.vptiles_test_data, get_vptiles))


def load_xyz_test_data(xyz_test_data):
    try:
        xyz_test_data = pickle.loads(xyz_file.read_bytes())
    except FileNotFoundError:
        xyz_file.write_bytes(pickle.dumps(xyz_test_data))
    return xyz_test_data


def load_vp_img_test_data(vp_img):
    try:
        vp_img = pickle.loads(vp_img_file.read_bytes())
    except FileNotFoundError:
        vp_img_file.write_bytes(pickle.dumps(vp_img))
    return vp_img


def load_vptiles_test_data(vptiles):
    try:
        vptiles = pickle.loads(vptiles_file.read_bytes())
    except FileNotFoundError:
        vptiles_file.write_bytes(pickle.dumps(vptiles))
    return vptiles
