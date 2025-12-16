from pathlib import Path

import numpy as np

from py360tools import CMP, ERP, Projection, Tile


class SMSE:
    proj_obj: Projection
    proj_obj_ref: Projection

    def __init__(self, projection: str, proj_shape: tuple, sph_file: str):
        self.projection = projection
        self.proj_shape = proj_shape

        self.make_sph_points_mask(Path(sph_file))

    def smse_nn(self, tile1: Tile, tile2: Tile) -> float:
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param tile1: The original image
        :param tile2: The image degraded
        :return:
        """
        x1, y1 = tile1.position
        x2, y2 = tile1.position + tile1.shape
        tile_mask = self.sph_points_mask[y1:y2, x1:x2]

        tile_1_m = tile1.canvas * tile_mask
        tile_2_m = tile2.canvas * tile_mask

        sqr_dif = (tile_1_m - tile_2_m) ** 2

        smse_nn = sqr_dif.sum() / np.sum(tile_mask)
        return smse_nn

    sph_points_mask: np.ndarray

    def make_sph_points_mask(self, sph_file: Path):
        """
        Processes the spherical file to identify points on the unit sphere (r=1)
        based on given elevation and azimuth coordinates. Marks these points on
        the faces of the specified projection type (CMP or ERP). The projection
        resolution must match the one used in videos, for example, (2160, 3240)
        for CMP. The resulting map is saved and used as a mask to determine which
        pixels are taken into account during the calculation of the s-MSE metric.

        :param sph_file: The spherical file containing elevation and azimuth
                         coordinates to process
        :type sph_file: pathlib.Path
        :return: np.ndarray
        """
        """
        O arquivo txt contem as coordenadas como ea (elevação, azimute).
        Vamos identificar pontos na esfera (r=1) com essas coordenadas e marcar nas
        faces da projeção estes pontos. A projeção deve ter a mesma resolução dos
        vídeos, (2160, 3240) para CMP. Este mapa é guardado e usado como máscara
        para selecionar quais pixeis devem ser considerados no cálculo do s-mse.
        """
        sph_file_lines = sph_file.read_text().splitlines()[1:]
        ea_array = np.array(list(map(self.lines_2_list, sph_file_lines))).T

        if self.projection == 'cmp':
            nm = CMP.ea2nm_face(ea=ea_array, proj_shape=self.proj_shape)[0]
        elif self.projection == 'erp':
            nm = ERP.ea2nm(ea=ea_array, proj_shape=self.proj_shape)[0]
        else:
            nm = np.ndarray([])
        self.sph_points_mask = np.zeros(self.proj_shape)
        self.sph_points_mask[nm[0], nm[1]] = 1
        return self.sph_points_mask

    @staticmethod
    def lines_2_list(line) -> list:
        strip_line = line.strip()
        split_line = strip_line.split()
        map_float_line = map(float, split_line)
        map_rad_line = map(np.deg2rad, map_float_line)
        return list(map_rad_line)
