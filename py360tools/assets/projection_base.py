from abc import ABC, abstractmethod

import numpy as np

from py360tools.assets.tiling import Tiling
from py360tools.utils.lazyproperty import LazyProperty
from py360tools.utils.util_transform import create_nm_coords
from .canvas import Canvas


class ProjectionBase(ABC):
    def __init__(self, *, proj_res, tiling='1x1'):
        """

        :param proj_res: A string representing the projection resolution. E.g. '600x3000'
        :type proj_res: str
        :param tiling: A string representing the tiling. E.g. '1x1' or '3x2'
        :type tiling: str
        """
        # Build frame
        self.canvas = Canvas(proj_res)

        # Build tiling
        self.tiling = Tiling(tiling, self)

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

    def __str__(self):
        return f'{self.__class__.__name__}({self.canvas})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.canvas}@{self.tiling})'
