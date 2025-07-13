import unittest
from pathlib import Path

import numpy as np

from py360tools import Viewport
from py360tools.draw import show
from py360tools.transform import ea2xyz
from py360tools.utils import load_test_data

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent
__ASSETS__ = __PATH__ / f'assets/{__FILENAME__.stem}'
__ASSETS__.mkdir(parents=True, exist_ok=True)


class TestViewport(unittest.TestCase):
    viewport: Viewport

    @classmethod
    def setUpClass(cls):
        cls.viewport = Viewport(resolution='440x360', fov='110x90')
        cls.viewport.yaw_pitch_roll = (np.deg2rad(-0), np.deg2rad(0), np.deg2rad(0))

    def test_is_viewport(self):
        is_viewport_test_file = Path(f'{__ASSETS__}/TestViewport_is_viewport_test_file.pickle')
        point = np.array([[0, 50],
                          [0, 60],
                          [0, -50],
                          [0, -60],
                          [40, 0],
                          [50, 0],
                          [-40, 0],
                          [-50, 0],
                          ]).T
        point = np.deg2rad(point)
        point = ea2xyz(ea=point)

        is_viewport = self.viewport.is_viewport(point)
        is_viewport_test = load_test_data(is_viewport_test_file, is_viewport)
        self.assertTrue(np.array_equal(is_viewport, is_viewport_test))

    def test_make_vp_xyz_base(self):
        xyz_default_test_file = Path(f'{__ASSETS__}/TestViewport_xyz_default_test_file.pickle')
        xyz_default = load_test_data(xyz_default_test_file, self.viewport._xyz_default)
        self.assertTrue(np.array_equal(xyz_default, self.viewport._xyz_default))

    def test_make_vp_xyz_rotated(self):
        vp_xyz_rotated_test_file = Path(f'{__ASSETS__}/TestViewport_vp_xyz_rotated_test_file.pickle')
        vp_xyz_rotated = load_test_data(vp_xyz_rotated_test_file, self.viewport.xyz)
        self.assertTrue(np.array_equal(vp_xyz_rotated, self.viewport.xyz))

    def test_normals_default(self):
        normals_default_test_file = Path(f'{__ASSETS__}/TestViewport_normals_default_test_file.pickle')
        normals_default = load_test_data(normals_default_test_file, self.viewport.normals)
        self.assertTrue(np.array_equal(self.viewport.normals, normals_default))

    def test_normals_rotated(self):
        normals_rotated_test_file = Path(f'{__ASSETS__}/TestViewport_normals_rotated_test_file.pickle')
        normals_rotated = load_test_data(normals_rotated_test_file, self.viewport.normals)
        self.assertTrue(np.array_equal(self.viewport.normals, normals_rotated))

    # def test_is_viewport(self):
    #     pass
