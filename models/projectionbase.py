from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np

from models.tiling import Tiling, Tile
from models.viewport import Viewport
from utils.util import splitx


class ProjectionInterface(ABC):
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


class ProjectionBase(ProjectionInterface, ABC):
    def __init__(self, *, proj_res, tiling='1x1'):
        """

        :param proj_res: A string representing the projection resolution. e.g. '600x3000'
        :type proj_res: str
        :param tiling: A string representing the tiling. e.g. '1x1' or '3x2'
        :type tiling: str
        """
        self.name = self.__class__.__name__

        self.proj_res = proj_res
        self.tiling = Tiling(tiling, self)

        # About projection
        self.shape = np.array(splitx(self.proj_res)[::-1], dtype=int)
        self.coord_nm = np.array(np.mgrid[0:self.shape[0], 0:self.shape[1]])
        self.coord_xyz = self.nm2xyz(self.coord_nm, self.shape)

    def extract_viewport(self, viewport, frame_img):
        """

        :param viewport:
        :type viewport: Viewport
        :param frame_img:
        :type frame_img: np.ndarray
        :return:
        :type:
        """

        nm_coord = self.xyz2nm(viewport.vp_xyz_rotated, self.shape)
        nm_coord = nm_coord.transpose((1, 2, 0))
        vp_img = cv2.remap(frame_img,
                           map1=nm_coord[..., 1:2].astype(np.float32),
                           map2=nm_coord[..., 0:1].astype(np.float32),
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
        # show1(vp_img)
        return vp_img

    def get_vptiles(self, viewport):
        """

        :param viewport:
        :type viewport: Viewport
        :return: Return a list with all the tiles used in the viewport.
        :rtype: list[Tile]
        """
        if str(self.tiling) == '1x1': return [self.tiling.tile_list[0]]

        vptiles = []
        for tile in self.tiling.tile_list:
            borders_xyz = self.nm2xyz(tile.borders, self.shape)
            if viewport.is_viewport(borders_xyz):
                vptiles.append(tile)
        return vptiles

    @property
    def tile_list(self):
        return self.tiling.tile_list
