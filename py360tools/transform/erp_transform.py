import numpy as np


def erp2vu(*, nm: np.ndarray, proj_shape=None) -> np.ndarray:
    if proj_shape is None:
        proj_shape = nm.shape[1:]

    shape = [2]
    for i in range(len(nm.shape) - 1):
        shape.append(1)

    n1 = np.asarray([0.5, 0.5]).reshape(shape)
    n2 = np.asarray([proj_shape[0], proj_shape[1]]).reshape(shape)

    vu = (nm + n1) / n2
    return vu


def vu2ea(*, vu: np.ndarray) -> np.ndarray:
    shape = [2]
    for i in range(len(vu.shape) - 1):
        shape.append(1)

    n1 = np.asarray([-np.pi, 2 * np.pi]).reshape(shape)
    n2 = np.asarray([np.pi / 2, -np.pi]).reshape(shape)

    ea = (vu * n1) + n2
    return ea


def ea2vu(*, ea) -> np.ndarray:
    """

    :param ea: shape==(2,...)
    :return:
    """

    vu = np.zeros(ea.shape)
    vu[0] = -ea[0] / np.pi + 0.5
    vu[1] = ea[1] / (2 * np.pi) + 0.5
    return vu


def vu2erp(*, vu, proj_shape=None) -> np.ndarray:
    if proj_shape is None:
        proj_shape = vu.shape[1:]

    shape = np.ones([len(vu.shape)])
    shape[0] = 2

    n1 = np.asarray([proj_shape[0], proj_shape[1]]).reshape(shape.astype(int))

    nm = vu * (n1 - 1)
    nm = nm + 0.5
    return nm.astype(int)


def ea2erp(*, ea: np.ndarray, proj_shape=None) -> np.ndarray:
    """

    :param ea: in rad
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (m, n) pixel coord using nearest neighbor
    """
    from lib.transform.transform import normalize_ea
    ea = normalize_ea(ea=ea)
    vu = ea2vu(ea=ea)
    nm = vu2erp(vu=vu, proj_shape=proj_shape)
    return nm


def erp2ea(*, nm: np.ndarray, proj_shape=None) -> np.ndarray:
    vu = erp2vu(nm=nm, proj_shape=proj_shape)
    ea = vu2ea(vu=vu)
    return ea
