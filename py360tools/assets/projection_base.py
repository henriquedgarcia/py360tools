from abc import ABC, abstractmethod

import numpy as np

from py360tools.assets.tiling import Tiling
from py360tools.assets.viewport import Viewport
from py360tools.utils.lazyproperty import LazyProperty
from py360tools.utils.util_transform import create_nm_coords
from .canvas import Canvas
from ..transform.transform import get_vptiles
from ..utils.util_transform import extract_viewport


class ProjectionBase(ABC):
    def __init__(self, *, proj_res, tiling='1x1', vp_res='1x1', fov_res='1x1'):
        """

        :param proj_res: A string representing the projection resolution. e.g. '600x3000'
        :type proj_res: str
        :param tiling: A string representing the tiling. e.g. '1x1' or '3x2'
        :type tiling: str
        :param vp_res:
        :type vp_res: str
        :param fov_res:
        :type fov_res: str
        """
        # Build frame
        self.canvas = Canvas(proj_res)

        # Build tiling
        self.tiling = Tiling(tiling, self)

        # Build viewport
        self.viewport = Viewport(resolution=vp_res,
                                 fov=fov_res)

    @abstractmethod
    def nm2xyz(self, nm) -> np.ndarray:
        """
        Projection specific.

        :param nm: shape==(2,...)
        :type nm: np.ndarray
        :return:
        """
        pass

    @abstractmethod
    def xyz2nm(self, xyz) -> np.ndarray:
        """
        Projection specific.

        :param xyz: shape==(2,...)
        :type xyz: np.ndarray
        :return:
        """
        pass

    @LazyProperty
    def nm(self):
        return create_nm_coords(self.canvas.shape)

    @LazyProperty
    def xyz(self):
        return self.nm2xyz(self.nm)

    @property
    def yaw_pitch_roll(self):
        return self.viewport.yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value):
        self.viewport.yaw_pitch_roll = value

    def extract_viewport(self, frame_array, yaw_pitch_roll=None):
        if yaw_pitch_roll is not None:
            self.viewport.yaw_pitch_roll = yaw_pitch_roll
        return extract_viewport(self, self.viewport, frame_array)

    def get_vptiles(self, yaw_pitch_roll=None):
        """

        :return: Return a list with all the tiles used in the viewport.
        :rtype: list[Tile]
        """
        if yaw_pitch_roll is not None:
            self.viewport.yaw_pitch_roll = yaw_pitch_roll
        vptiles = get_vptiles(self, self.viewport)
        return vptiles

    def __str__(self):
        return f'{self.__class__.__name__}({self.canvas})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.canvas}@{self.tiling})'
