from abc import ABC, abstractmethod
from typing import Optional, Generator, Any

import cv2
import numpy as np

from models.tiling import Tiling, Tile
from models.viewport import Viewport
from utils.lazyproperty import LazyProperty
from utils.util import get_tile_borders
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


class ProjectionBuilder:
    def _build_projection(self, proj_res: str):
        self.proj_res = proj_res
        self.shape = np.array(splitx(self.proj_res)[::-1], dtype=int)

    def _build_tiling(self, tiling: str):
        self.tiling = Tiling(tiling, self.proj_res)

    def _build_tile_list(self):
        self.tile_shape = (self.shape / self.tiling.shape).astype(int)
        self.tile_list = (self._build_tile(tile_id) for tile_id in range(self.tiling.ntiles))

    def _build_tile(self, tile_id):
        tile = Tile(tile_id, str(self.tiling))
        tile.shape = self.tile_shape
        tile.borders = get_tile_borders(tile_id, self.tiling.shape, self.tile_shape)
        tile.borders_xyz = self.nm2xyz(tile.borders)
        tile.position_nm = tile.borders[::, 0]
        return tile

    def _build_viewport(self, vp_shape, fov):
        self.viewport = Viewport(vp_shape=vp_shape, fov=fov)


class ProjectionBase(ProjectionBuilder, ProjectionInterface, ABC):
    proj_res: str
    shape: np.ndarray
    tiling: Tiling
    tile_list: list[Tile]
    viewport: Optional[Viewport] = None
    tile_shape: np.ndarray
    tile_list: Generator[Tile, Any, None]

    def __init__(self, *, proj_res, tiling='1x1', vp_shape=None, fov=None):
        """

        :param proj_res: A string representing the projection resolution. e.g. '600x3000'
        :type proj_res: str
        :param tiling: A string representing the tiling. e.g. '1x1' or '3x2'
        :type tiling: str
         :param vp_shape:
        :type vp_shape: np.ndarray
        :param fov:
        :type fov: np.ndarray
        """

        self._build_projection(proj_res)
        self._build_tiling(tiling)
        self._build_tile_list()

        if None not in [vp_shape, fov]:
            self._build_viewport(vp_shape, fov)

    @LazyProperty
    def coord_nm(self):
        return np.array(np.mgrid[0:self.shape[0], 0:self.shape[1]])

    @LazyProperty
    def coord_xyz(self):
        return self.nm2xyz(self.coord_nm)

    @property
    def yaw_pitch_roll(self):
        return self.viewport.yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value):
        self.viewport.yaw_pitch_roll = value

    def extract_viewport(self, frame_img):
        """

        :param frame_img:
        :type frame_img: np.ndarray
        :return:
        :type:
        """
        if self.viewport is None:
            raise ValueError('Viewport not defined during instantiation. It is necessary to fill in the "vp_shape" and '
                             '"fov" parameters.')

        nm_coord = self.xyz2nm(self.viewport.vp_xyz_rotated)
        nm_coord = nm_coord.transpose((1, 2, 0))
        vp_img = cv2.remap(frame_img,
                           map1=nm_coord[..., 1:2].astype(np.float32),
                           map2=nm_coord[..., 0:1].astype(np.float32),
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
        # show1(vp_img)
        return vp_img

    def get_vptiles(self):
        """

        :return: Return a list with all the tiles used in the viewport.
        :rtype: list[Tile]
        """
        if str(self.tiling) == '1x1': return [tile for tile in self.tile_list]

        vptiles = []
        for tile in self.tile_list:
            if self.viewport.is_viewport(tile.borders_xyz):
                vptiles.append(tile)
        return vptiles

    def __str__(self):
        return self.__class__.__name__
