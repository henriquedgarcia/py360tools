from abc import ABC, abstractmethod
from typing import Union, Callable

import numpy as np

from models.tiling import Tiling, Tile
from utils.util import get_borders_value, show, splitx
from utils.viewport import Viewport


class Attributes:
    canvas: np.ndarray
    frame_img: np.ndarray  # = np.zeros([0])
    n_tiles: int
    nm2xyz: Callable
    proj_coord_xyz: np.ndarray
    proj_shape: np.ndarray
    tile_border_base: np.ndarray
    tile_borders_nm: np.ndarray
    tile_borders_xyz: list
    tile_position_list: list
    tile_shape: Union[np.ndarray, tuple]
    tiling: Tiling
    viewport: Viewport
    xyz2nm: Callable
    yaw_pitch_roll: np.ndarray


class ViewportMethods(Attributes):
    def get_viewport_image(self, frame_img, yaw_pitch_roll=None) -> np.ndarray:
        """

        :param frame_img: A full frame of the projection
        :type frame_img: np.ndarray
        :param yaw_pitch_roll: A tuple or ndarray with the yaw, pitch and roll angles of the viewport
        :type yaw_pitch_roll: np.ndarray | tuple
        :return:
        """
        if yaw_pitch_roll is not None:
            self.yaw_pitch_roll = yaw_pitch_roll

        out = self.viewport.get_vp(frame_img, self.xyz2nm)
        return out


class TilesMethods(Attributes):
    def get_tiles_position_nm(self, tile_id):
        """

        :param tile_id: tile index value
        :type tile_id: int | str
        :return: A ndarray with the tile position in projection
        :rtype: np.ndarray
        """
        return self.tiling.tiles[int(tile_id)].pixel_position

    def get_tile_borders_nm(self, tile_id):
        """
        :param tile_id: tile index value
        :type tile_id: int | str
        :return: A ndarray with shape == (ntiles, 2), with the tile borders in (n, m) coords
        :rtype: np.ndarray
        """
        return self.tile_border_base + self.get_tiles_position_nm(tile_id)

    def get_tile_borders_xyz(self, tile_id):
        """
        :param tile_id: tile index value
        :type tile_id: int | str
        :return: A ndarray with shape == (ntiles, 3), with the tile borders in (x, y, z) coords
        :rtype: np.ndarray
        """
        return self.nm2xyz(nm=self.get_tile_borders_nm(tile_id), proj_shape=self.proj_shape)

    def get_vptiles(self, yaw_pitch_roll=None):
        """

        :param yaw_pitch_roll: yaw, pitch and roll angles of the viewport
        :type yaw_pitch_roll: np.ndarray | tuple
        :return:
        :rtype: list[Tile]
        """
        if self.tiling == '1x1': return [self.tiling.tiles[0]]

        if yaw_pitch_roll is not None:
            self.yaw_pitch_roll = yaw_pitch_roll

        vptiles = []
        for tile in self.tiling.tiles:
            if self.viewport.is_viewport(self.get_tile_borders_xyz(tile.tile_id)):
                vptiles.append(tile)
        return vptiles


class DrawMethods(TilesMethods, Attributes):
    def draw_tile_border(self, idx, lum=255) -> np.ndarray:
        """
        Do not return copy
        :param idx:
        :type idx: int
        :param lum:
        :return:
        """
        canvas = np.zeros(self.proj_shape, dtype='uint8')
        canvas[self.get_tile_borders_nm(idx)] = lum
        return canvas

    def draw_all_tiles_borders(self, lum=255):
        canvas = np.zeros(self.proj_shape, dtype='uint8')
        for tile in self.tiling.tiles:
            canvas = canvas + self.draw_tile_border(idx=int(tile), lum=lum)
        return canvas

    def draw_vp_tiles(self, lum=255):
        canvas = np.zeros(self.proj_shape, dtype='uint8')
        for tile in self.get_vptiles():
            canvas = canvas + self.draw_tile_border(idx=int(tile), lum=lum)
        return canvas

    def draw_vp_mask(self, lum=255) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        canvas = np.zeros(self.proj_shape, dtype='uint8')

        inner_prod = np.tensordot(self.viewport.rotated_normals.T, self.proj_coord_xyz, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        canvas[belong] = lum

        return canvas

    def draw_vp_borders(self, thickness=1, lum=255):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """
        canvas = np.zeros(self.proj_shape, dtype='uint8')

        vp_borders_xyz = get_borders_value(array=self.viewport.vp_xyz_rotated, thickness=thickness)
        nm = self.xyz2nm(vp_borders_xyz, proj_shape=self.proj_shape).astype(int)
        canvas[nm[0, ...], nm[1, ...]] = lum
        return canvas


class Props(Attributes):
    proj_coord_nm: Union[list, np.ndarray]

    @property
    def yaw_pitch_roll(self) -> np.ndarray:
        return self.viewport.yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: Union[np.ndarray, list]):
        self.viewport.yaw_pitch_roll = np.array(value)


class ProjBase(Props,
               DrawMethods,
               TilesMethods,
               ViewportMethods,
               ABC):
    def __init__(self, *, proj_res: str, fov: str, tiling: str = '1x1', vp_shape: Union[np.ndarray, tuple, list] = None):
        """

        @param proj_res: A string representing the projection resolution. e.g. '600x3000'
        @param fov: A string representing the fov. e.g. '100x90'
        @param tiling: A string representing the tiling. e.g. '1x1' or '3x2
        @param vp_shape: A tuple, or nd.array or list with shape of viewport. e.g. (300, 600) - (height, width)
        """
        # About projection
        self.proj_res = proj_res
        self.proj_shape = np.array(splitx(self.proj_res)[::-1], dtype=int)
        self.proj_h = self.proj_shape[0]
        self.proj_w = self.proj_shape[1]
        self.proj_coord_nm = np.mgrid[0:self.proj_h, 0:self.proj_w]
        self.proj_coord_xyz = self.nm2xyz(self.proj_coord_nm, self.proj_shape)

        # About Tiling
        self.tiling = tiling
        self.tiling_shape = np.array(splitx(self.tiling)[::-1], dtype=int)
        self.tiling_h = self.tiling_shape[0]
        self.tiling_w = self.tiling_shape[1]

        # About Tiles
        self.n_tiles = self.tiling_h * self.tiling_w
        self.tile_shape = (self.proj_shape / self.tiling_shape).astype(int)
        self.tile_h = self.tile_shape[0]
        self.tile_w = self.tile_shape[1]
        self.tile_position_list = self.get_tiles_position_nm()
        self.tile_border_base = get_borders_value(array=np.mgrid[0:self.tile_shape[0], 0:self.tile_shape[1]])
        self.tile_borders_nm = self.get_tile_borders_nm()
        self.tile_borders_xyz = self.get_tile_borders_xyz()

        # About Viewport
        self.fov = fov
        self.fov_shape = np.deg2rad(splitx(self.fov)[::-1])
        if vp_shape is None:
            vp_shape = np.round(self.fov_shape * self.proj_shape[0] / 4).astype('int')
        self.vp_shape = np.asarray(vp_shape)
        self.viewport = Viewport(self.vp_shape, self.fov_shape)

        self.yaw_pitch_roll = [0, 0, 0]

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
