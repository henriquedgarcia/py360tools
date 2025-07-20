import numpy as np

from py360tools.assets.projection_base import ProjectionBase
from py360tools.transform.transform import ea2xyz, xyz2ea
from py360tools.utils.util_transform import check_ea


class ERP(ProjectionBase):
    def nm2xyz(self, nm):
        vu = ERP.nm2vu(nm=nm, proj_shape=self.shape)
        ea = ERP.vu2ea(vu=vu)
        xyz = ea2xyz(ea=ea)  # common
        return xyz

    def xyz2nm(self, xyz):
        ea = xyz2ea(xyz=xyz)
        vu = ERP.ea2vu(ea=ea)
        nm = ERP.vu2nm(vu=vu, proj_shape=self.shape)
        return nm

    @staticmethod
    def nm2vu(*, nm: np.ndarray, proj_shape=None) -> np.ndarray:
        if proj_shape is None:
            proj_shape = nm.shape[1:]

        shape = [2]
        for i in range(len(nm.shape) - 1):
            shape.append(1)

        n1 = np.asarray([0.5, 0.5]).reshape(shape)
        n2 = np.asarray([proj_shape[0], proj_shape[1]]).reshape(shape)

        vu = (nm + n1) / n2
        return vu

    @staticmethod
    def vu2ea(*, vu: np.ndarray) -> np.ndarray:
        shape = [2]
        for i in range(len(vu.shape) - 1):
            shape.append(1)

        n1 = np.asarray([-np.pi, 2 * np.pi]).reshape(shape)
        n2 = np.asarray([np.pi / 2, -np.pi]).reshape(shape)

        ea = (vu * n1) + n2
        return ea

    @staticmethod
    def ea2vu(*, ea) -> np.ndarray:
        """

        :param ea: shape==(2,...)
        :return:
        """

        vu = np.zeros(ea.shape)
        vu[0] = -ea[0] / np.pi + 0.5
        vu[1] = ea[1] / (2 * np.pi) + 0.5
        return vu

    @staticmethod
    def vu2nm(*, vu, proj_shape=None) -> np.ndarray:
        if proj_shape is None:
            proj_shape = vu.shape[1:]

        shape = np.ones([len(vu.shape)])
        shape[0] = 2

        n1 = np.asarray([proj_shape[0], proj_shape[1]]).reshape(shape.astype(int))

        nm = vu * (n1 - 1)
        nm = nm + 0.5
        return nm.astype(int)

    @staticmethod
    def ea2nm(*, ea: np.ndarray, proj_shape=None) -> np.ndarray:
        """

        :param ea: in rad
        :param proj_shape: shape of projection in numpy format: (height, width)
        :return: (m, n) pixel coord using the nearest neighbor
        """
        ea = check_ea(ea=ea)
        vu = ERP.ea2vu(ea=ea)
        nm = ERP.vu2nm(vu=vu, proj_shape=proj_shape)
        return nm

    @staticmethod
    def nm2ea(*, nm: np.ndarray, proj_shape=None) -> np.ndarray:
        vu = ERP.nm2vu(nm=nm, proj_shape=proj_shape)
        ea = ERP.vu2ea(vu=vu)
        return ea
