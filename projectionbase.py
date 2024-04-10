from numpy import linalg
from abc import ABC, abstractmethod
from math import pi
from typing import Union

import cv2
import numpy as np
from viewport import Viewport
from util import splitx, get_borders, show1


class ProjProps(ABC):
    proj_res: str
    tiling: str
    fov: str
    vp_rotated_xyz: np.ndarray
    frame_img = np.zeros([0])

    @abstractmethod
    def nm2xyz(self, nm_coord: np.ndarray, shape: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def xyz2nm(self, xyz_coord: np.ndarray, shape: Union[np.ndarray, tuple], round_nm: bool) -> np.ndarray:
        pass

    # <editor-fold desc="About the Projection">
    _proj_shape: np.ndarray = None

    @property
    def proj_shape(self) -> np.ndarray:
        if self._proj_shape is None:
            self._proj_shape = np.array(splitx(self.proj_res)[::-1], dtype=int)
        return self._proj_shape

    _proj_h: int = None

    @property
    def proj_h(self) -> int:
        if not self._proj_h:
            self._proj_h = self.proj_shape[0]
        return self._proj_h

    _proj_w: int = None

    @property
    def proj_w(self) -> int:
        if not self._proj_w:
            self._proj_w = self.proj_shape[1]
        return self._proj_w

    _projection: np.ndarray = None

    @property
    def projection(self) -> np.ndarray:
        if not self._projection:
            self._projection = np.zeros(self.proj_shape, dtype='uint8')
        return self._projection

    @projection.setter
    def projection(self, value):
        self._projection = value

    _proj_coord_nm: np.ndarray = None

    @property
    def proj_coord_nm(self) -> np.ndarray:
        if not self._proj_coord_nm:
            self._proj_coord_nm = np.mgrid[range(self.proj_h), range(self.proj_w)]
        return self._proj_coord_nm

    _proj_coord_xyz: np.ndarray = None

    @property
    def proj_coord_xyz(self) -> np.ndarray:
        if not self._proj_coord_xyz:
            self._proj_coord_xyz = self.nm2xyz(self.proj_coord_nm, self.proj_shape)
        return self._proj_coord_xyz

    # </editor-fold>

    # <editor-fold desc="About Tiling">
    _tiling_shape: np.ndarray = None

    @property
    def tiling_shape(self) -> np.ndarray:
        if self._tiling_shape is None:
            self._tiling_shape = np.array(splitx(self.tiling)[::-1], dtype=int)
        return self._tiling_shape

    _tiling_h: int = None

    @property
    def tiling_h(self) -> int:
        if not self._tiling_h:
            self._tiling_h = self.tiling_shape[0]
        return self._tiling_h

    _tiling_w: int = None

    @property
    def tiling_w(self) -> int:
        if not self.tiling_w:
            self._tiling_w = self.tiling_shape[1]
        return self.tiling_w

    # </editor-fold>

    # <editor-fold desc="About Tiles">
    _n_tiles: int = None

    @property
    def n_tiles(self) -> int:
        if not self._n_tiles:
            self._n_tiles = self.tiling_shape[0] * self.tiling_shape[1]
        return self._n_tiles

    _tile_shape: np.ndarray = None

    @property
    def tile_shape(self) -> np.ndarray:
        if self._tile_shape is None:
            self._tile_shape = (self.proj_shape / self.tiling_shape).astype(int)
        return self._tile_shape

    _tile_h: int = None

    @property
    def tile_h(self) -> int:
        if not self._tile_h:
            self._tile_h = self.tile_shape[0]
        return self._tile_h

    _tile_w: int = None

    @property
    def tile_w(self) -> int:
        if not self._tile_w:
            self._tile_w = self.tile_shape[1]
        return self._tile_w

    _tile_position_list: np.ndarray = None

    @property
    def tile_position_list(self) -> np.ndarray:
        """
        top-left pixel position
        :return: (N,2)
        """
        if self._tile_position_list is None:
            tile_position_list = []
            for n in range(0, self.proj_shape[0], self.tile_shape[0]):
                for m in range(0, self.proj_shape[1], self.tile_shape[1]):
                    tile_position_list.append((n, m))
            self._tile_position_list = np.array(tile_position_list)
        return self._tile_position_list

    _tile_border_base: np.ndarray = None

    @property
    def tile_border_base(self) -> np.ndarray:
        """

        :return: shape==(2, 2*(tile_height+tile_weight)
        """
        if self._tile_border_base is None:
            self._tile_border_base = get_borders(shape=self.tile_shape)
        return self._tile_border_base

    _tile_borders_nm: np.ndarray = None

    @property
    def tile_borders_nm(self) -> np.ndarray:
        """

        :return:
        """
        # projection agnostic
        if self._tile_borders_nm is None:
            _tile_borders_nm = []
            for tile in range(self.n_tiles):
                tile_position = self.tile_position_list[tile].reshape(2, -1)
                _tile_borders_nm.append(self.tile_border_base + tile_position)
            self._tile_borders_nm = np.array(_tile_borders_nm)
        return self._tile_borders_nm

    _tile_borders_xyz: list = None

    @property
    def tile_borders_xyz(self) -> list:
        """
        shape = (3, H, W) "WxH array" OR (3, N) "N points (z, y, x)"
        :return:
        """
        if not self._tile_borders_xyz:
            self._tile_borders_xyz = []
            for tile in range(self.n_tiles):
                borders_nm = self.tile_borders_nm[tile]
                borders_xyz = self.nm2xyz(nm_coord=borders_nm,
                                          shape=self.proj_shape)
                self._tile_borders_xyz.append(borders_xyz)
        return self._tile_borders_xyz

    # </editor-fold>

    # <editor-fold desc="About Viewport">
    _fov_shape: np.ndarray = None

    @property
    def fov_shape(self) -> np.ndarray:
        if self._fov_shape is None:
            self._fov_shape = np.deg2rad(splitx(self.fov)[::-1])
        return self._fov_shape

    _vp_shape: np.ndarray = None

    @property
    def vp_shape(self) -> np.ndarray:
        if self._vp_shape is None:
            self._vp_shape = np.round(self.fov_shape * self.proj_shape / (pi, 2 * pi)).astype('int')
        return self._vp_shape

    @vp_shape.setter
    def vp_shape(self, value: np.ndarray):
        self._vp_shape = value

    _viewport: Viewport = None

    @property
    def viewport(self) -> Viewport:
        if not self._viewport:
            self._viewport = Viewport(self.vp_shape, self.fov_shape)
        return self._viewport

    _vp_image: np.ndarray = None

    @property
    def vp_image(self) -> np.ndarray:
        nm_coord = self.xyz2nm(self.vp_rotated_xyz,
                               self.frame_img.shape,
                               round_nm=False)
        nm_coord = nm_coord.transpose((1, 2, 0))
        out = cv2.remap(self.frame_img,
                        map1=nm_coord[..., 1:2].astype(np.float32),
                        map2=nm_coord[..., 0:1].astype(np.float32),
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)

        self._vp_image = out
        return self._vp_image

    # </editor-fold>

    # <editor-fold desc="About Position">
    @property
    def yaw_pitch_roll(self) -> np.ndarray:
        return self.viewport.yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: Union[np.ndarray, list]):
        self.viewport.yaw_pitch_roll = np.array(value)

    # </editor-fold>


class Draw(ProjProps, ABC):
    def show(self):
        show1(self.projection)

    ##############################################
    # Draw methods
    def draw_all_tiles_borders(self, lum=255):
        self.clear_projection()
        for tile in range(self.n_tiles):
            self.draw_tile_border(idx=int(tile), lum=lum)
        return self.projection

    @abstractmethod
    def get_vptiles(self):
        ...

    def draw_vp_tiles(self, lum=255):
        self.clear_projection()
        for tile in self.get_vptiles():
            self.draw_tile_border(idx=int(tile), lum=lum)
        return self.projection

    def draw_tile_border(self, idx, lum=255):
        n, m = self.tile_borders_nm[idx]
        self.projection[n, m] = lum

    def draw_vp_mask(self, lum=255) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        self.clear_projection()
        rotated_normals = self.viewport.rotated_normals.T
        inner_product = np.tensordot(rotated_normals, self.proj_coord_xyz, axes=1)
        belong = np.all(inner_product <= 0, axis=0)
        self.projection[belong] = lum
        return self.projection

    def draw_vp_borders(self, lum=255, thickness=1):
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """
        self.clear_projection()

        vp_borders_xyz = get_borders(coord_nm=self.viewport.vp_rotated_xyz, thickness=thickness)

        nm_coord = self.xyz2nm(vp_borders_xyz, shape=self.proj_shape, round_nm=True).astype(int)
        self.projection[nm_coord[0, ...], nm_coord[1, ...]] = lum
        return self.projection

    def clear_projection(self):
        self.projection = None


class ProjBase(Draw, ProjProps, ABC):
    def __init__(self, *, proj_res: str, tiling: str,
                 fov: str, vp_shape: np.ndarray = None):
        self.proj_res = proj_res
        self.tiling = tiling
        self.fov = fov
        self.vp_shape = vp_shape
        self.yaw_pitch_roll = [0, 0, 0]

    def get_vptiles(self, yaw_pitch_roll=None) -> list[str]:
        """

        :return:
        """
        if self.tiling == '1x1': return ['0']
        if yaw_pitch_roll is not None:
            self.yaw_pitch_roll = yaw_pitch_roll
        vptiles = [str(tile) for tile in range(self.n_tiles)
                   if self.viewport.is_viewport(self.tile_borders_xyz[tile])]
        return vptiles

    def get_viewport(self, frame_img: np.ndarray, yaw_pitch_roll=None) -> np.ndarray:
        # todo: não implementado???
        if yaw_pitch_roll is not None:
            self.yaw_pitch_roll = yaw_pitch_roll
        self.frame_img = frame_img
        return self.vp_image

    @staticmethod
    def ea2xyz(ae) -> np.ndarray:
        """
        Convert from horizontal coordinate system  in radians to cartesian system.
        ISO/IEC JTC1/SC29/WG11/N17197l: Algorithm descriptions of projection format
        conversion and video quality metrics in 360Lib Version 5.

        :param np.ndarray ae:
            In Rad. Shape == (2, ...)
        :return: (x, y, z)
        """
        newshape = (3,) + ae.shape[1:]
        xyz = np.zeros(newshape)
        xyz[0] = np.cos(ae[0]) * np.sin(ae[1])
        xyz[1] = -np.sin(ae[0])
        xyz[2] = np.cos(ae[0]) * np.cos(ae[1])
        xyz_r = np.round(xyz, 6)
        return xyz_r

    @staticmethod
    def xyz2ea(xyz: np.ndarray) -> np.ndarray:
        """
        Convert from cartesian system to horizontal coordinate system in radians
        :param xyz: shape = (3, ...)
        :return: np.ndarray([azimuth, elevation]) - in rad. shape = (2, ...)
        """
        new_shape = (2,) + xyz.shape[1:]
        ea = np.zeros(new_shape)
        # z-> x,
        r = linalg.norm(xyz, axis=0)
        ea[0] = np.arcsin(-xyz[1] / r)
        ea[1] = np.arctan2(xyz[0], xyz[2])
        return ea

