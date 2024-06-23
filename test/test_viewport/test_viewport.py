import unittest
from pathlib import Path

import numpy as np

from models import Viewport
from utils.util import show

show = show
__FILENAME__ = Path(__file__).absolute()
__PATH__ = __FILENAME__.parent


class TestViewport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viewport = Viewport(vp_shape=np.ndarray([500, 600]),
                                fov=np.ndarray([110, 90]))

    def test_make_normals_base(self):
        self.viewport.base_normals == self.viewport.base_normals
        pass

    def test_make_vp_xyz_base(self):
        pass

    def test_is_viewport(self):
        pass

    def test_get_vp_borders_xyz(self):
        pass
