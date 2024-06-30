import numpy as np

from utils.util import show, get_borders_value
from models.projectionbase import ProjectionBase


class DrawHelper:
    @staticmethod
    def show(value):
        show(value)

    @staticmethod
    def draw_tile_border(*, projection: ProjectionBase, idx, lum=255) -> np.ndarray:
        """

        :param projection:
        :param idx:
        :type idx: int
        :param lum:
        :return:
        """
        canvas = np.zeros(projection.shape, dtype='uint8')
        canvas[projection.tile_list[idx].borders[0], projection.tile_list[idx].borders[1]] = lum
        return canvas

    @classmethod
    def draw_all_tiles_borders(cls, *, projection: ProjectionBase, lum=255):
        """

        :param projection:
        :param lum:
        :return:
        """
        canvas = np.zeros(projection.shape, dtype='uint8')
        for tile in projection.tile_list:
            canvas = canvas + cls.draw_tile_border(projection=projection, idx=int(tile), lum=lum)
        return canvas

    def draw_vp_tiles(self, *, projection: ProjectionBase, lum=255):
        """

        :param projection:
        :param lum:
        :return:
        """
        canvas = np.zeros(projection.shape, dtype='uint8')
        for tile in projection.vptiles:
            canvas = canvas + self.draw_tile_border(projection=projection, idx=int(tile), lum=lum)
        return canvas

    @staticmethod
    def draw_vp_mask(*, projection: ProjectionBase, lum=255) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param projection:
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        canvas = np.zeros(projection.shape, dtype='uint8')
        belong = projection.viewport.is_viewport(projection.coord_xyz)
        canvas[belong] = lum

        return canvas

    @staticmethod
    def draw_vp_borders(*, projection: ProjectionBase, thickness=1, lum=255):
        """
        Project the sphere using ERP. Where is Viewport the
        :param projection:
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """
        canvas = np.zeros(projection.shape, dtype='uint8')

        vp_borders_xyz = get_borders_value(array=projection.viewport.vp_xyz_rotated, thickness=thickness)
        nm = projection.xyz2nm(vp_borders_xyz).astype(int)
        canvas[nm[0, ...], nm[1, ...]] = lum
        return canvas


draw = DrawHelper()
