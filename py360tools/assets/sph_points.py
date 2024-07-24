import numpy as np
from pathlib import Path

from py360tools.transform.cmp_transform import ea2nm_face
from py360tools.transform.erp_transform import ea2nm


class SpherePoints:
    sph_points_mask: np.ndarray

    def __init__(self, sph_file: Path, video_shape: tuple, proj: str):
        self.sph_points_mask = np.zeros(video_shape)
        sph_file_lines = sph_file.read_text().splitlines()[1:]
        # for each line (sample), convert to cartesian system and horizontal system
        for line in sph_file_lines:
            el, az = list(map(np.deg2rad, map(float, line.strip().split())))  # to rad
            ea = np.array([[az], [el]])

            if proj == 'erp':
                m, n = ea2nm(ea=ea, proj_shape=video_shape)
            elif proj == 'cmp':
                (m, n), face = ea2nm_face(ea=ea, proj_shape=video_shape)
            else:
                raise ValueError(f'wrong value to {proj=}')

            self.sph_points_mask[n, m] = 1
