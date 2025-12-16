import numpy as np
from skimage.metrics import structural_similarity

from py360tools import Projection


class SSIM:
    @staticmethod
    def ssim(img1, img2):
        return structural_similarity(img1, img2, data_range=255.0, gaussian_weights=True,
                                     sigma=1.5, use_sample_covariance=False)

    def tiles_ssim(self, proj_obj: Projection, proj_obj_ref: Projection):
        tiles_ssim_list = []
        assert proj_obj.tiling == proj_obj_ref.tiling, 'The projections must have the same tiling'
        for tile, tile_ref in zip(proj_obj.tile_list, proj_obj_ref.tile_list):
            if tile.path is None or tile_ref.path is None: continue
            tiles_ssim = self.ssim(tile.canvas, tile_ref.canvas, data_range=255.0, gaussian_weights=True,
                                   sigma=1.5, use_sample_covariance=False)
            tiles_ssim_list.append(tiles_ssim)

        tiles_ssim_avg = float(np.average(tiles_ssim_list))
        return tiles_ssim_avg
