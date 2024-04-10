from math import pi

import numpy as np

from projectionbase import ProjBase


class ERP(ProjBase):
    def nm2xyz(self, nm_coord: np.ndarray, shape: np.ndarray):
        """
        ERP specific.

        :param nm_coord: shape==(2,...)
        :param shape: (N, M)
        :return:
        """
        azimuth = ((nm_coord[1] + 0.5) / shape[1] - 0.5) * 2 * np.pi
        elevation = ((nm_coord[0] + 0.5) / shape[0] - 0.5) * -np.pi

        z = np.cos(elevation) * np.cos(azimuth)
        y = -np.sin(elevation)
        x = np.cos(elevation) * np.sin(azimuth)

        xyz_coord = np.array([x, y, z])
        return xyz_coord

    def xyz2nm(self, xyz_coord: np.ndarray, shape: np.ndarray = None, round_nm: bool = False):
        """
        ERP specific.

        :param xyz_coord: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param shape: the shape of projection that cover all sphere
        :param round_nm: round the coords? is not needed.
        :return:
        """
        if shape is None:
            shape = xyz_coord.shape[:2]

        proj_h, proj_w = shape[:2]

        r = np.sqrt(np.sum(xyz_coord ** 2, axis=0))

        elevation = np.arcsin(xyz_coord[1] / r)
        azimuth = np.arctan2(xyz_coord[0], xyz_coord[2])

        v = elevation / pi + 0.5
        u = azimuth / (2 * pi) + 0.5

        n = v * proj_h - 0.5
        m = u * proj_w - 0.5

        if round_nm:
            n = np.mod(np.round(n), proj_h)
            m = np.mod(np.round(m), proj_w)

        return np.array([n, m])

    @staticmethod
    def erp2vu(nm: np.ndarray, shape=None) -> np.ndarray:
        if shape is None:
            shape = nm.shape[1:]
        vu = (nm + [[[0.5]], [[0.5]]]) / [[[shape[0]]], [[shape[1]]]]
        return vu

    @staticmethod
    def vu2ea(vu: np.ndarray) -> np.ndarray:
        ea = (vu * [[[-np.pi]], [[2 * np.pi]]]) + [[[np.pi / 2]], [[-np.pi]]]
        # ea = (vu - [[[0.5]], [[0.5]]]) * [[[-np.pi]], [[2 * np.pi]]]
        return ea

    @staticmethod
    def erp2hcs(nm: np.ndarray, shape=None) -> np.ndarray:
        vu = erp2vu(nm, shape=shape)
        ea = vu2ea(vu)
        return ea

    @staticmethod
    def erp2xyz(nm: np.ndarray, shape=None) -> np.ndarray:
        """
        ERP specific.

        :param nm: [(n, m], ...]
        :param shape: (H, W)
        :return:
        """
        ea = erp2hcs(nm, shape=shape)
        xyz = ea2xyz(ea)
        return xyz

    @staticmethod
    def xyz2erp(xyz, shape=None) -> np.ndarray:
        ea = xyz2ea(xyz)
        nm = ea2erp(ea, shape)
        return nm

    @staticmethod
    def ea2erp(ea: np.ndarray, shape=None) -> np.ndarray:
        """

        :param ea: in rad
        :param shape: shape of projection in numpy format: (height, width)
        :return: (m, n) pixel coord using nearest neighbor
        """
        ea = normalize_ea(ea)
        vu = ea2vu(ea)
        nm = vu2erp(vu, shape)
        return nm

    @staticmethod
    def normalize_ea(ea):
        _90_deg = np.pi / 2
        _180_deg = np.pi
        _360_deg = 2 * np.pi

        # if pitch>90
        sel = ea[1] > _90_deg
        ea[0, sel] = _180_deg - ea[0, sel]
        ea[1, sel] = ea[1, sel] + _180_deg

        # if pitch<90
        sel = ea[1] < -_90_deg
        ea[0, sel] = -_180_deg - ea[0, sel]
        ea[1, sel] = ea[1, sel] + _180_deg

        # if yaw>=180 or yaw<180
        sel = ea[1] >= _180_deg or ea[1] < -_180_deg
        ea[1, sel] = (ea[1, sel] + _180_deg) % _360_deg - _180_deg

        return ea

    @staticmethod
    def ea2vu(ea):
        vu = np.zeros(ea)
        vu[0] = -ea[0] / np.pi + 0.5
        vu[1] = ea[1] / (2 * np.pi) + 0.5
        return vu

    @staticmethod
    def vu2erp(vu, shape=None):
        if shape is None:
            shape = vu.shape[1:]

        nm = vu * [[[shape[0]]], [[shape[1]]]]
        nm = np.ceil(nm)
        return nm
