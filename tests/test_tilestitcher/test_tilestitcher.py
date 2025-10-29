import unittest
from pathlib import Path

import numpy as np

from py360tools import Viewport, ERP, ProjectionBase, TileStitcher, Tile
from py360tools.draw import show
from py360tools.utils import load_test_data

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__MEDIA__ = Path(f'tests/input/12x8')  # 'tile{tile}.mp4' de 0 a 95 (tile=range(96))
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}/'
__ASSETS__.mkdir(parents=True, exist_ok=True)

frame_erp_test_data_file = Path(f'{__ASSETS__}/frame_erp_test_data.pickle')
frame_cmp_test_data_file = Path(f'{__ASSETS__}/frame_cmp_test_data.pickle')


class TestTileStitcher(unittest.TestCase):
    projection: ProjectionBase

    @classmethod
    def setUpClass(cls):
        # erp '144x72', '288x144','432x216','576x288'
        # cmp '144x96', '288x192','432x288','576x384'

        cls.erp = ERP(tiling='12x8', proj_res='576x288')
        cls.cmp = ERP(tiling='12x8', proj_res='576x384')

        cls.yaw_pitch_roll = np.deg2rad((90, 0, 0))
        cls.viewport_erp = Viewport('440x360', '110x90', cls.erp)
        cls.viewport_cmp = Viewport('440x360', '110x90', cls.cmp)

    def test_tile_stitcher_cmp(self):
        # inicializar
        tiles_seen_cmp: list[Tile]
        tiles_seen_cmp = self.viewport_cmp.get_vptiles(self.yaw_pitch_roll)
        for tile in tiles_seen_cmp:
            tile.path = str(__MEDIA__ / f'erp/tile{tile.idx}.mp4')
        tile_stitcher_cmp = TileStitcher(tiles_seen_cmp, self.cmp, gray=True)

        # criar iterador
        tile_stitcher_cmp_iter = iter(tile_stitcher_cmp)

        # testar next
        frame_cmp1 = next(tile_stitcher_cmp_iter)
        self.frame_cmp_test_data = load_test_data(frame_cmp_test_data_file, frame_cmp1)
        self.assertTrue(np.array_equal(frame_cmp1, self.frame_cmp_test_data))
        frame_cmp2 = next(tile_stitcher_cmp_iter)
        self.assertFalse(np.array_equal(frame_cmp2, self.frame_cmp_test_data))

        # testar reset
        tile_stitcher_cmp.reset()
        frame_cmp1 = next(tile_stitcher_cmp_iter)
        self.assertTrue(np.array_equal(frame_cmp1, self.frame_cmp_test_data))
        frame_cmp2 = next(tile_stitcher_cmp_iter)
        self.assertFalse(np.array_equal(frame_cmp2, self.frame_cmp_test_data))

    def test_tile_stitcher_erp(self):
        # inicializar
        tiles_seen_erp: list[Tile]
        tiles_seen_erp = self.viewport_erp.get_vptiles(self.yaw_pitch_roll)
        for tile in tiles_seen_erp:
            tile.path = str(__MEDIA__ / f'erp/tile{tile.idx}.mp4')
        tile_stitcher_erp = TileStitcher(tiles_seen_erp, self.erp, gray=True)

        # criar iterador
        tile_stitcher_erp_iter = iter(tile_stitcher_erp)

        # testar next
        frame_erp1 = next(tile_stitcher_erp_iter)
        self.frame_erp_test_data = load_test_data(frame_erp_test_data_file, frame_erp1)
        self.assertTrue(np.array_equal(frame_erp1, self.frame_erp_test_data))
        frame_erp2 = next(tile_stitcher_erp_iter)
        self.assertFalse(np.array_equal(frame_erp1, frame_erp2))
        # show(frame_erp1)

        # testar reset
        tile_stitcher_erp.reset()
        frame_erp1 = next(tile_stitcher_erp_iter)
        self.assertTrue(np.array_equal(frame_erp1, self.frame_erp_test_data))
        frame_erp2 = next(tile_stitcher_erp_iter)
        self.assertFalse(np.array_equal(frame_erp1, frame_erp2))
