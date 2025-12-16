import numpy as np

from py360tools import Projection, Tile


class WSMSE:
    proj_obj: Projection
    proj_obj_ref: Projection

    def __init__(self, projection: str, proj_shape: tuple):
        self.projection = projection
        self.proj_shape = proj_shape

        self.make_weight_ndarray_dict()

    def tiles_wsmse(self, proj_obj: Projection, proj_obj_ref: Projection):
        self.proj_obj = proj_obj
        self.proj_obj_ref = proj_obj_ref

        tiles_wsmse_list = []
        assert proj_obj.tiling == proj_obj_ref.tiling, 'The projections must have the same tiling'
        for tile, tile_ref in zip(proj_obj.tile_list, proj_obj_ref.tile_list):
            if tile.path is None or tile_ref.path is None: continue
            tiles_wsmse = self.wsmse(tile, tile_ref)
            tiles_wsmse_list.append(tiles_wsmse)

        tiles_wsmse_avg = float(np.average(tiles_wsmse_list))
        return tiles_wsmse_avg

    def wsmse(self, tile1: Tile, tile2: Tile) -> float:
        """
        Must be the same size
        :param tile1:
        :param tile2:
        :return:
        """
        x1, y1 = tile1.position
        x2, y2 = tile1.position + tile1.shape
        weight_tile = self.weight_ndarray[y1:y2, x1:x2]

        wmse = np.sum(weight_tile * (tile1.canvas - tile2.canvas) ** 2) / np.sum(weight_tile)
        return wmse

    weight_ndarray: np.ndarray

    def make_weight_ndarray_dict(self):
        h, w = self.proj_shape
        if self.projection == 'erp':
            pi_proj = np.pi / h
            proj_h_2 = 0.5 - h / 2

            def func(y, x):
                weights = np.cos((y + proj_h_2) * pi_proj)
                return weights

        elif self.projection == 'cmp':
            a = h / 2
            r = a / 2
            r1 = 0.5 - r
            r2 = r ** 2

            def func(y, x):
                x = x % a
                y = y % a
                d = (x + r1) ** 2 + (y + r1) ** 2
                weights = (1 + d / r2) ** (-1.5)
                return weights
        else:
            raise ValueError(f'Unknown projection: {self.projection}')
        self.weight_ndarray = np.fromfunction(func, (h, w), dtype=float)
        # plt.imshow(np.ones((cmp_h, cmp_w)) * 255 * np.fromfunction(func_cmp, (cmp_h, cmp_w), dtype=float));plt.show()
        # plt.imshow(np.ones((erp_h, erp_w)) * 255 * np.fromfunction(func_erp, (erp_h, erp_w), dtype=float));plt.show()
