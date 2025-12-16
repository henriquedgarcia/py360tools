import numpy as np
from skimage.metrics import mean_squared_error

from py360tools import Projection


class MSE:
    @staticmethod
    def mse(img1, img2):
        return mean_squared_error(img1, img2)

    def tiles_mse(self, proj_obj: Projection, proj_obj_ref: Projection):
        tiles_mse_list = []
        assert proj_obj.tiling == proj_obj_ref.tiling, 'The projections must have the same tiling'
        for tile, tile_ref in zip(proj_obj.tile_list, proj_obj_ref.tile_list):
            if tile.path is None or tile_ref.path is None: continue
            tiles_mse = self.mse(tile.canvas, tile_ref.canvas)
            tiles_mse_list.append(tiles_mse)

        tiles_mse_avg = float(np.average(tiles_mse_list))
        return tiles_mse_avg
