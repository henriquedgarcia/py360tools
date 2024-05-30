from typing import Union

import numpy as np

from .projectionbase import ProjBase, ea2xyz, xyz2ea, normalize_ea


# from PIL import Image


# from .util import compose


class ERP(ProjBase):
    def nm2xyz(self,
               nm: np.ndarray,
               proj_shape: Union[np.ndarray, tuple]
               ) -> np.ndarray:
        if proj_shape is None:
            proj_shape = nm.shape[1:]

        vu = erp2vu(nm=nm,
                    proj_shape=proj_shape)
        ea = vu2ea(vu=vu)
        xyz = ea2xyz(ea=ea)  # common
        return xyz

    def xyz2nm(self,
               xyz: np.ndarray,
               proj_shape: np.ndarray = None
               ) -> np.ndarray:
        """
        ERP specific.

        :param xyz: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param proj_shape: the shape of projection that cover all sphere. tuple as (N, M)
        :return:
        """
        if proj_shape is None:
            proj_shape = xyz.shape[:2]

        ea = xyz2ea(xyz=xyz)
        vu = ea2vu(ea=ea)
        nm = vu2erp(vu=vu,
                    proj_shape=proj_shape)

        return nm


def erp2vu(*,
           nm: np.ndarray,
           proj_shape=None
           ) -> np.ndarray:
    if proj_shape is None:
        proj_shape = nm.shape[1:]

    shape = [2]
    for i in range(len(nm.shape) - 1):
        shape.append(1)

    n1 = np.asarray([0.5, 0.5]).reshape(shape)
    n2 = np.asarray([proj_shape[0], proj_shape[1]]).reshape(shape)

    vu = (nm + n1) / n2
    return vu


def vu2ea(*,
          vu: np.ndarray
          ) -> np.ndarray:
    shape = [2]
    for i in range(len(vu.shape) - 1):
        shape.append(1)

    n1 = np.asarray([-np.pi, 2 * np.pi]).reshape(shape)
    n2 = np.asarray([np.pi / 2, -np.pi]).reshape(shape)

    ea = (vu * n1) + n2
    return ea


def ea2vu(*,
          ea
          ) -> np.ndarray:
    """

    :param ea: shape==(2,...)
    :return:
    """

    vu = np.zeros(ea.shape)
    vu[0] = -ea[0] / np.pi + 0.5
    vu[1] = ea[1] / (2 * np.pi) + 0.5
    return vu


def vu2erp(*,
           vu,
           proj_shape=None
           ) -> np.ndarray:
    if proj_shape is None:
        proj_shape = vu.shape[1:]

    shape = [2]
    for i in range(len(vu.shape) - 1):
        shape.append(1)

    n1 = np.asarray([proj_shape[0], proj_shape[1]]).reshape(shape)

    nm = vu * (n1 - 1)
    nm = np.floor(nm)
    return nm.astype(int)


def ea2erp(*,
           ea: np.ndarray,
           proj_shape=None
           ) -> np.ndarray:
    """

    :param ea: in rad
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (m, n) pixel coord using nearest neighbor
    """
    ea = normalize_ea(ea=ea)
    vu = ea2vu(ea=ea)
    nm = vu2erp(vu=vu,
                proj_shape=proj_shape)
    return nm


def erp2ea(*,
           nm: np.ndarray,
           proj_shape=None
           ) -> np.ndarray:
    vu = erp2vu(nm=nm,
                proj_shape=proj_shape)
    ea = vu2ea(vu=vu)
    return ea

# def test_erp():
#     # erp '144x72', '288x144','432x216','576x288'
#     yaw_pitch_roll = np.deg2rad((70, 0, 0))
#     height, width = 288, 576
#
#     ########################################
#     # Open Image
#     frame_img: Union[Image, list] = Image.open('images/erp1.jpg')
#     frame_img = frame_img.resize((width, height))
#
#     erp = ERP(tiling='6x4', proj_res=f'{width}x{height}', fov='100x90')
#     erp.yaw_pitch_roll = yaw_pitch_roll
#     compose(erp, frame_img)
#
#
# if __name__ == '__main__':
#     test_erp()
