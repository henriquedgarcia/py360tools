import numpy as np

from utils.util import show, get_borders_value


class DrawHelper:
    @staticmethod
    def show(value):
        show(value)

    @staticmethod
    def draw_tile_border(*, projection, idx, lum=255) -> np.ndarray:
        """

        :param projection:
        :param idx:
        :type idx: int
        :param lum:
        :return:
        """
        canvas = np.zeros(projection.projection, dtype='uint8')
        canvas[projection.get_tile_borders_nm(idx)[0], projection.get_tile_borders_nm(idx)[1]] = lum
        return canvas

    @classmethod
    def draw_all_tiles_borders(cls, *, projection, lum=255):
        """

        :param projection:
        :param lum:
        :return:
        """
        canvas = np.zeros(projection.projection, dtype='uint8')
        for tile in projection.tiling.tiles:
            canvas = canvas + cls.draw_tile_border(projection=projection, idx=int(tile), lum=lum)
        return canvas

    def draw_vp_tiles(self, *, projection, lum=255):
        """

        :param projection:
        :param lum:
        :return:
        """
        canvas = np.zeros(projection.projection, dtype='uint8')
        for tile in projection.get_vptiles():
            canvas = canvas + self.draw_tile_border(projection=projection, idx=int(tile), lum=lum)
        return canvas

    @staticmethod
    def draw_vp_mask(*, projection, lum=255) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param projection:
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        canvas = np.zeros(projection.projection, dtype='uint8')

        inner_prod = np.tensordot(projection.viewport.rotated_normals.T, projection.coord_xyz, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        canvas[belong] = lum

        return canvas

    @staticmethod
    def draw_vp_borders(*, projection, thickness=1, lum=255):
        """
        Project the sphere using ERP. Where is Viewport the
        :param projection:
        :param lum: value to draw line
        :param thickness: in pixel.
        :return: a numpy.ndarray with one deep color
        """
        canvas = np.zeros(projection.projection, dtype='uint8')

        vp_borders_xyz = get_borders_value(array=projection.viewport.vp_xyz_rotated, thickness=thickness)
        nm = projection.xyz2nm(vp_borders_xyz, proj_shape=projection.projection).astype(int)
        canvas[nm[0, ...], nm[1, ...]] = lum
        return canvas


draw = DrawHelper()
