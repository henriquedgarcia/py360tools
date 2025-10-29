import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from py360tools import Viewport, CMP
from py360tools.draw import show
from py360tools.utils import load_test_data

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}'
__ASSETS__.mkdir(parents=True, exist_ok=True)


class TestCmpDrawMethods(unittest.TestCase):
    projection: CMP
    frame_array: np.ndarray

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 384, 576

        cls.projection = CMP(tiling='6x4', proj_res=f'{width}x{height}')
        cls.viewport = Viewport('440x360',
                                '110x90', cls.projection)
        cls.viewport.yaw_pitch_roll = np.deg2rad((0, 0, 0))

        # Open Image
        frame_img: Image = Image.open('images/cmp1.png')
        frame_img = frame_img.resize((width, height))
        cls.frame_array = np.array(frame_img)

    def test_draw_all_tiles_borders(self):
        draw_all_tiles_borders_test_file = __ASSETS__ / 'draw_all_tiles_borders_test_file.pickle'
        canvas = self.projection.draw_all_tiles_borders()
        draw_all_tiles_borders_test = load_test_data(draw_all_tiles_borders_test_file,
                                                     canvas)
        # show(canvas)
        self.assertTrue(np.array_equal(canvas, draw_all_tiles_borders_test))

    def test_draw_vp_borders(self):
        draw_vp_borders_test_file = Path(f'{__ASSETS__}/draw_vp_borders_test_file.pickle')
        canvas = self.viewport.draw_borders()
        draw_vp_borders_test = load_test_data(draw_vp_borders_test_file, canvas)
        # show(canvas)
        self.assertTrue(np.array_equal(canvas, draw_vp_borders_test))

    def test_draw_vp_mask(self):
        draw_vp_mask_test_file = Path(f'{__ASSETS__}/draw_vp_mask_test_file.pickle')
        canvas = self.viewport.draw_mask()
        draw_vp_mask_test = load_test_data(draw_vp_mask_test_file, canvas)
        # show(canvas)
        self.assertTrue(np.array_equal(canvas, draw_vp_mask_test))

    def test_draw_vp_tiles(self):
        draw_vp_tiles_test_file = Path(f'{__ASSETS__}/draw_vp_tiles_test_file.pickle')
        canvas = self.viewport.draw_tiles_seen()
        draw_vp_tiles_test = load_test_data(draw_vp_tiles_test_file,
                                            canvas)
        # show(canvas)
        self.assertTrue(np.array_equal(canvas, draw_vp_tiles_test))

    def test_get_viewport_image(self):
        get_viewport_image_test_file = Path(f'{__ASSETS__}/get_viewport_image_test_file.pickle')
        get_viewport_image = self.viewport.extract_viewport(self.frame_array)
        get_viewport_image_test = load_test_data(get_viewport_image_test_file,
                                                 get_viewport_image)
        # show(get_viewport_image)
        self.assertTrue(np.array_equal(get_viewport_image_test, get_viewport_image))
