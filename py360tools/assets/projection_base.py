from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Union

import numpy as np

from ..utils import create_nm_coords, splitx
from ..utils.util import unflatten_index, get_borders_coord_nm


@dataclass
class Tile:
    idx: int
    shape: Union[tuple, np.ndarray]
    position_tiling: tuple
    position_nm: np.ndarray
    borders_nm: np.ndarray
    borders_xyz: np.ndarray


class ProjectionBase(ABC):
    def __init__(self, *, proj_res, tiling='1x1'):
        """

        :param proj_res: A string representing the projection resolution. E.g. '600x300'
        :type proj_res: str
        :param tiling: A string representing the tiling. E.g. '1x1' or '3x2'
        :type tiling: str
        """
        # Build frame
        self.resolution = proj_res
        self.shape = np.array(splitx(self.resolution)[::-1], dtype=int)

        # Build tiling
        self.tiling: str = tiling
        self.tiling_shape = np.array(splitx(self.tiling)[::-1], dtype=int)
        self.n_tile = self.tiling_shape.prod()
        self.tile_shape = self.shape // self.tiling_shape

        self.tile_list = self.make_tile_list()

    def make_tile_list(self):
        tile_list: dict[int, Tile] = {}
        for tile in range(self.n_tile):
            tile_position_tiling = unflatten_index(tile, self.tiling_shape)[::-1]
            tile_position_nm = (tile_position_tiling * self.tile_shape).astype(int)
            tile_borders_nm = get_borders_coord_nm(tile_position_nm, self.tile_shape).astype(int)
            tile_borders_xyz = self.nm2xyz(tile_borders_nm)
            tile_list.update({tile: Tile(idx=tile,
                                         shape=self.tile_shape,
                                         position_tiling=tile_position_tiling,
                                         position_nm=tile_position_nm.astype(int),
                                         borders_nm=tile_borders_nm,
                                         borders_xyz=tile_borders_xyz)})
        return tile_list

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

    @cached_property
    def nm(self):
        return create_nm_coords(self.shape)

    @cached_property
    def xyz(self) -> np.ndarray:
        return self.nm2xyz(self.nm)

    def __str__(self):
        return f'{self.__class__.__name__}({self.resolution})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.resolution}@{self.tiling})'

    def draw_tile_border(self, tile: Tile, lum=255) -> np.ndarray:
        """
        Draw the borders of a tile in the canvas.
        :param tile:
        :param lum:
        :return:
        """
        canvas = np.zeros(self.shape, dtype='uint8')
        canvas[tile.borders_nm[0], tile.borders_nm[1]] = lum
        return canvas

    def draw_all_tiles_borders(self, lum=255):
        """
        Draw all borders of all tiles in the canvas.
        :param lum:
        :return:
        """
        canvas = np.zeros(self.shape, dtype='uint8')
        for tile in self.tile_list.values():
            canvas = canvas + self.draw_tile_border(tile=tile, lum=lum)
        return canvas
