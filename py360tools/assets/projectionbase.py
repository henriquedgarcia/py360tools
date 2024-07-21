from .projectionframe import ProjectionFrame
from py360tools.utils.util import create_nm_coords
from abc import ABC, abstractmethod

import cv2
import numpy as np

from py360tools.assets.tiling import Tiling
from py360tools.assets.viewport import Viewport
from py360tools.utils.lazyproperty import LazyProperty
from py360tools.utils.util import splitx


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
        self.frame = ProjectionFrame(proj_res)

        # Build tiling
        self.tiling = Tiling(tiling, self.frame)

        # Build viewport
        vp_shape = np.array(splitx(vp_res)[::-1])
        fov = np.deg2rad(np.array(splitx(fov_res)[::-1]))
        self.viewport = Viewport(vp_shape=vp_shape,
                                 fov=fov)

    def extract_viewport(self, frame_array):
        return extract_viewport(self, self.viewport, frame_array)

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
    def xyz2nm(self, xyz: np.ndarray) -> np.ndarray:
        """
        Projection specific.

        :param xyz: shape==(2,...)
        :type xyz: np.ndarray
        :return:
        """
        pass

    @LazyProperty
    def nm(self):
        return create_nm_coords(self.frame.shape)

    @LazyProperty
    def xyz(self):
        return self.nm2xyz(self.nm)

    @property
    def vptiles(self):
        """

        :return: Return a list with all the tiles used in the viewport.
        :rtype: list[Tile]
        """
        if str(self.tiling) == '1x1': return [tile for tile in self.tiling.tile_list]

        vptiles = []
        for tile in self.tiling.tile_list:
            borders_xyz = self.nm2xyz(tile.borders_nm)
            if np.any(self.viewport.is_viewport(borders_xyz)):
                vptiles.append(tile)
        return vptiles

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f'{self.__class__.__name__}({self.frame.proj_res}@{self.tiling})'


def extract_viewport(projection, viewport, frame_array):
    """

    :param projection:
    :param viewport:
    :type viewport: Viewport
    :param frame_array:
    :type frame_array: np.ndarray
    :return:
    :type:
    """

    nm_coord = projection.xyz2nm(viewport.xyz)
    nm_coord = nm_coord.transpose((1, 2, 0))
    vp_img = cv2.remap(frame_array,
                       map1=nm_coord[..., 1:2].astype(np.float32),
                       map2=nm_coord[..., 0:1].astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_WRAP)
    # show(vp_img)
    return vp_img
