from pathlib import Path

import numpy as np

from py360tools import ERP
from py360tools import CMP


class SpherePoints:
    sph_points_mask: np.ndarray

    def __init__(self, sph_file: Path, video_shape: tuple, proj: str):
        self.sph_points_mask = np.zeros(video_shape)
        sph_file_lines = sph_file.read_text().splitlines()[1:]
        # for each line (sample), convert to a cartesian system and horizontal system
        for line in sph_file_lines:
            el, az = list(map(np.deg2rad, map(float, line.strip().split())))  # to rad
            ea = np.array([[az], [el]])

            if proj == 'erp':
                m, n = ERP.ea2nm(ea=ea, proj_shape=video_shape)
            elif proj == 'cmp':
                (m, n), face = CMP.ea2nm_face(ea=ea, proj_shape=video_shape)
            else:
                raise ValueError(f'wrong value to {proj=}')

            self.sph_points_mask[n, m] = 1
