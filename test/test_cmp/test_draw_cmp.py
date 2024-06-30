from utils.util import load_test_data
import pickle
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from helper.draw import draw
from models import CMP, Viewport
from utils.util import show

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / 'assets'

draw_all_tiles_borders_test_file = Path(f'{__ASSETS__}/TestCmpDrawMethods_draw_all_tiles_borders_test_file.pickle')
draw_vp_borders_test_file = Path(f'{__ASSETS__}/TestCmpDrawMethods_draw_vp_borders_test_file.pickle')
draw_vp_mask_test_file = Path(f'{__ASSETS__}/TestCmpDrawMethods_draw_vp_mask_test_file.pickle')
draw_vp_tiles_test_file = Path(f'{__ASSETS__}/TestCmpDrawMethods_draw_vp_tiles_test_file.pickle')


class TestCmpDrawMethods(unittest.TestCase):
    projection: CMP
    viewport: Viewport
    frame_array: np.ndarray

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 384, 576

        cls.projection = CMP(tiling='6x4', proj_res=f'{width}x{height}',
                             vp_shape=np.array([360, 440]),
                             fov=np.array([np.deg2rad(90), np.deg2rad(110)]))
        cls.projection.yaw_pitch_roll = np.deg2rad((0, 0, 0))

        # Open Image
        frame_img: Image = Image.open('images/cmp1.png')
        frame_img = frame_img.resize((width, height))
        cls.frame_array = np.array(frame_img)

    def test_draw_all_tiles_borders(self):
        draw_all_tiles_borders = draw.draw_all_tiles_borders(projection=self.projection)
        draw_all_tiles_borders_test = load_test_data(draw_all_tiles_borders_test_file,
                                                     draw_all_tiles_borders)
        # show(draw_all_tiles_borders)
        self.assertTrue(np.array_equal(draw_all_tiles_borders_test, draw_all_tiles_borders))

    def test_draw_vp_tiles(self):
        draw_vp_tiles = draw.draw_vp_tiles(projection=self.projection)
        draw_vp_tiles_test = load_test_data(draw_vp_borders_test_file,
                                            draw_vp_tiles)

        self.assertTrue(np.array_equal(draw_vp_tiles, draw_vp_tiles_test))

    def test_draw_vp_borders(self):
        draw_vp_borders = draw.draw_vp_borders(projection=self.projection)
        draw_vp_borders_test = load_test_data(draw_vp_borders)
        show(draw_vp_borders)
        self.assertTrue(np.array_equal(draw_vp_borders_test, draw_vp_borders))

    def test_draw_vp_mask(self):
        draw_vp_mask = draw.draw_vp_mask(projection=self.projection)
        draw_vp_mask_test = load_draw_vp_mask(draw_vp_mask)

        self.assertTrue(np.array_equal(draw_vp_mask_test, draw_vp_mask))



