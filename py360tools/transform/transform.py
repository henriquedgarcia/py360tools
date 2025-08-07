import numpy as np
import pandas as pd
from numpy.linalg import norm

from py360tools.assets.matrot import MatRot


def ea2xyz(*, ea: np.ndarray) -> np.ndarray:
    """
    Convert from a horizontal coordinate system in radians to a cartesian system.
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
    Convert from a cartesian system to a horizontal coordinate system in radians
    :param xyz: shape = (3, ...)
    :type xyz: np.ndarray
    :return: np.ndarray([azimuth, elevation]) - in rad. Shape = (2, ...)
    :rtype: np.ndarray
    """
    ea = np.zeros((2,) + xyz.shape[1:])

    r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)

    ea[0] = np.arcsin(-xyz[1] / r)
    ea[1] = np.arctan2(xyz[0], xyz[2])
    ea[1] = (ea[1] + np.pi) % (2 * np.pi) - np.pi

    return ea


def position2displacement(df_positions):
    """
    Converts a position to a trajectory dataframe. The positions should have the 2 columns:
        - [0] yaw (float) in rads
        - [1] pitch (float) in rads

    The two columns of the dataframe will be converted into a numpy array.

    :param df_positions: Contains the position of the center of the viewport.
    :type df_positions: Pd.DataFrame
    :return : the displacement of the center of viewport by frame in radians
    :rtype: pd.DataFrame
    """

    # convert to numpy
    positions = np.array(df_positions[['yaw', 'pitch']])
    pos = ea2xyz(ea=positions.T).T

    # Calculate angle displacement = arc_cos(dot(v1, v2))
    dot_product = [np.sum(pos[i] * pos[i + 1] / (norm(pos[i]) * norm(pos[i + 1]))) for i in range(len(pos) - 1)]
    inst_angle = np.arccos(dot_product)

    return pd.DataFrame(inst_angle, columns=['displacement'])


def rotate(xyz, yaw_pitch_roll):
    matrix = MatRot.get_matrix(yaw_pitch_roll)
    return np.tensordot(matrix, xyz, axes=1)


def get_vptiles(projection, viewport) -> list:
    """

    :param projection:
    :param viewport:
    :return:
    :rtype: list[Tile]
    """
    if str(projection.tiling) == '1x1': return [0]

    vptiles = []
    for tile in projection.tile_list:
        borders_xyz = projection.nm2xyz(tile.borders)
        if np.any(viewport.is_viewport(borders_xyz)):
            vptiles.append(tile)
    return vptiles
