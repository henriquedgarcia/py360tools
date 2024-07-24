import numpy as np


def ea2xyz(*, ea: np.ndarray) -> np.ndarray:
    """
    Convert from horizontal coordinate system  in radians to cartesian system.
    ISO/IEC JTC1/SC29/WG11/N17197l: Algorithm descriptions of projection format conversion and video quality metrics in
    360Lib Version 5
    :param np.ndarray ea: In Rad. Shape == (2, ...)
    :return: (x, y, z)
    """
    new_shape = (3,) + ea.shape[1:]
    xyz = np.zeros(new_shape)
    xyz[0] = np.cos(ea[0]) * np.sin(ea[1])
    xyz[1] = -np.sin(ea[0])
    xyz[2] = np.cos(ea[0]) * np.cos(ea[1])
    return xyz


def xyz2ea(*, xyz):
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param xyz: shape = (3, ...)
    :type xyz: np.ndarray
    :return: np.ndarray([azimuth, elevation]) - in rad. shape = (2, ...)
    :rtype: np.ndarray
    """
    ea = np.zeros((2,) + xyz.shape[1:])

    r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)

    ea[0] = np.arcsin(-xyz[1] / r)
    ea[1] = np.arctan2(xyz[0], xyz[2])
    ea[1] = (ea[1] + np.pi) % (2 * np.pi) - np.pi

    return ea



