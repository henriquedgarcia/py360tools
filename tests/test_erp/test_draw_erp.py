import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import lib.helper.draw as draw
from lib.models.erp import ERP
from lib.utils.util import load_test_data

__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}'
__ASSETS__.mkdir(parents=True, exist_ok=True)


class TestErpDrawMethods(unittest.TestCase):
    projection: ERP

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'
        height, width = 288, 576

        cls.projection = ERP(tiling='6x4', proj_res=f'{width}x{height}', vp_res='440x360', fov_res='110x90')
        cls.projection._yaw_pitch_roll = np.deg2rad((70, 0, 0))

        # Open Image
        frame_img: Image = Image.open('images/erp1.png')
        frame_img = frame_img.resize((width, height))
        cls.frame_array = np.array(frame_img)

    def test_draw_all_tiles_borders(self):
        draw_all_tiles_borders_test_file = Path(f'{__ASSETS__}/draw_all_tiles_borders_test_file.pickle')
        draw_all_tiles_borders = draw.draw_all_tiles_borders(projection=self.projection)

        draw_all_tiles_borders_test = load_test_data(draw_all_tiles_borders_test_file,
                                                     draw_all_tiles_borders)
        # draw.show(draw_all_tiles_borders)
        self.assertTrue(np.array_equal(draw_all_tiles_borders_test, draw_all_tiles_borders))

    def test_draw_vp_borders(self):
        draw_vp_borders_test_file = Path(f'{__ASSETS__}/draw_vp_borders_test_file.pickle')
        draw_vp_borders = draw.draw_vp_borders(projection=self.projection)

        draw_vp_borders_test = load_test_data(draw_vp_borders_test_file,
                                              draw_vp_borders)
        # draw.show(draw_vp_borders)
        self.assertTrue(np.array_equal(draw_vp_borders, draw_vp_borders_test))

    def test_draw_vp_mask(self):
        draw_vp_mask_test_file = Path(f'{__ASSETS__}/draw_vp_mask_test_file.pickle')
        draw_vp_mask = draw.draw_vp_mask(projection=self.projection)

        draw_vp_mask_test = load_test_data(draw_vp_mask_test_file,
                                           draw_vp_mask)
        # draw.show(draw_vp_mask)
        self.assertTrue(np.array_equal(draw_vp_mask_test, draw_vp_mask))

    def test_draw_vp_tiles(self):
        draw_vp_tiles_test_file = Path(f'{__ASSETS__}/draw_vp_tiles_test_file.pickle')
        draw_vp_tiles = draw.draw_vp_tiles(projection=self.projection)
        draw_vp_tiles_test = load_test_data(draw_vp_tiles_test_file,
                                            draw_vp_tiles)
        # draw.show(draw_vp_tiles)
        self.assertTrue(np.array_equal(draw_vp_tiles_test, draw_vp_tiles))

    def test_get_vptiles(self):
        get_vptiles_test_file = Path(f'{__ASSETS__}/get_vptiles_test_file.pickle')
        get_vptiles = list(map(int, self.projection.vptiles))
        get_vptiles_test = load_test_data(get_vptiles_test_file,
                                          get_vptiles)
        # print(get_vptiles)
        self.assertTrue(np.array_equal(get_vptiles_test, get_vptiles))

    def test_get_viewport_image(self):
        get_viewport_image_test_file = Path(f'{__ASSETS__}/get_viewport_image_test_file.pickle')
        get_viewport_image = self.projection.extract_viewport(self.frame_array)
        get_viewport_image_test = load_test_data(get_viewport_image_test_file,
                                                 get_viewport_image)
        # draw.show(get_viewport_image)
        self.assertTrue(np.array_equal(get_viewport_image_test, get_viewport_image))
