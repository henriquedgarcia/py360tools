from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np

from models.tiling import Tiling, Tile
from models.viewport import Viewport
from utils.util import splitx


class ProjectionInterface(ABC):
    @abstractmethod
    def nm2xyz(self, nm: np.ndarray, proj_shape: np.ndarray) -> np.ndarray:
        """
        Projection specific.

        :param nm: shape==(2,...)
        :param proj_shape:
        :return:
        """
        pass

    @abstractmethod
    def xyz2nm(self, xyz: np.ndarray, proj_shape: Union[np.ndarray, tuple]) -> np.ndarray:
        """
        Projection specific.

        :param xyz: shape==(2,...)
        :param proj_shape:
        :return:
        """
        pass

    @abstractmethod
    def extract_viewport(self, viewport, frame_img, yaw_pitch_roll):
        """

        :param viewport:
        :param frame_img: A full frame of the projection
        :type frame_img: np.ndarray
        :type yaw_pitch_roll: np.ndarray | tuple
        :return:
        """
        pass

    @abstractmethod
    def get_vptiles(self, viewport):
        """

        :param viewport:
        :type viewport: Viewport
        :return: Return a list with all the tiles used in the viewport.
        :rtype: list[Tile]
        """


class ProjBase(ProjectionInterface, ABC):
    def __init__(self, *, proj_res, tiling='1x1'):
        """

        :param proj_res: A string representing the projection resolution. e.g. '600x3000'
        :type proj_res: str
        :param tiling: A string representing the tiling. e.g. '1x1' or '3x2'
        :type tiling: str
        """
        self.proj_res = proj_res
        self.name = self.__class__.__name__

        # About projection
        self.shape = np.array(splitx(self.proj_res)[::-1], dtype=int)
        self.coord_nm = np.array(np.mgrid[0:self.shape[0], 0:self.shape[1]])
        self.coord_xyz = self.nm2xyz(self.coord_nm, self.shape)

        self.tiling = Tiling(tiling, self)

    def extract_viewport(self, viewport, frame_img, yaw_pitch_roll=None):
        """

        :param viewport:
        :type viewport: Viewport
        :param frame_img:
        :type frame_img: np.ndarray
        :param yaw_pitch_roll:
        :type yaw_pitch_roll: np.ndarray | tuple
        :return:
        :type:
        """
        viewport.yaw_pitch_roll = yaw_pitch_roll

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
        if str(self.tiling) == '1x1': return [self.tiling.tiles[0]]

        vptiles = []
        for tile in self.tiling.tiles:
            borders_xyz = self.nm2xyz(tile.borders, self.shape)
            if viewport.is_viewport(borders_xyz):
                vptiles.append(tile)
        return vptiles
