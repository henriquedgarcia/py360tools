import numpy as np


def normalize_ea(*, ea):
    _90_deg = np.pi / 2
    _180_deg = np.pi
    _360_deg = 2 * np.pi

    new_ea = np.zeros(ea.shape)
    new_ea[0] = -np.abs(np.abs(ea[0] + _90_deg) - _180_deg) + _90_deg
    new_ea[1] = (ea[1] + _180_deg) % _360_deg - _180_deg

    return new_ea


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


def check_deg(axis_name, value):
    """

    :param axis_name:
    :type axis_name: str
    :param value: in rad
    :type value: float
    :return:
    :rtype: float
    """
    n_value = None
    if axis_name == 'azimuth':
        if value >= np.pi or value < -np.pi:
            n_value = (value + np.pi) % (2 * np.pi)
            n_value = n_value - np.pi
        return n_value
    elif axis_name == 'elevation':
        if value > np.pi / 2:
            n_value = 2 * np.pi - value
        elif value < -np.pi / 2:
            n_value = -2 * np.pi - value
        return n_value
    else:
        raise ValueError('"axis_name" not exist.')


# def rot_matrix(yaw_pitch_roll):
#     """
#     Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
#     See Wikipedia. Use:
#         X axis point to right
#         Y axis point to down
#         Z axis point to front
#
#     Examples
#     --------
#     >> x, y, z = point
#     >> mat = rot_matrix(yaw, pitch, roll)
#     >> mat @ (x, y, z)
#
#     :param yaw_pitch_roll: the rotation (yaw, pitch, roll) in rad.
#     :type yaw_pitch_roll: np.ndarray | list
#     :return: A 3x3 matrix of rotation for (z,y,x) vector
#     :rtype: np.ndarray
#     """
#     cos_rot = np.cos(yaw_pitch_roll)
#     sin_rot = np.sin(yaw_pitch_roll)
#
#     # pitch
#     mat_x = np.array([[1, 0, 0], [0, cos_rot[1], -sin_rot[1]], [0, sin_rot[1], cos_rot[1]]])
#     # yaw
#     mat_y = np.array([[cos_rot[0], 0, sin_rot[0]], [0, 1, 0], [-sin_rot[0], 0, cos_rot[0]]])
#     # roll
#     mat_z = np.array([[cos_rot[2], -sin_rot[2], 0], [sin_rot[2], cos_rot[2], 0], [0, 0, 1]])
#
#     return mat_y @ mat_x @ mat_z
#
#
# def rotate(xyz, yaw_pitch_roll):
#     mat_rot = rot_matrix(yaw_pitch_roll)
#     vp_xyz_rotated = np.tensordot(mat_rot, xyz, axes=1)


class MatRot:
    @staticmethod
    def rot_matrix(yaw_pitch_roll):
        """
        Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
        See Wikipedia. Use:
            X axis point to right
            Y axis point to down
            Z axis point to front

        Examples
        --------
        >> x, y, z = point
        >> mat = rot_matrix(yaw, pitch, roll)
        >> mat @ (x, y, z)

        :param yaw_pitch_roll: the rotation (yaw, pitch, roll) in rad.
        :type yaw_pitch_roll: np.ndarray | list
        :return: A 3x3 matrix of rotation for (z,y,x) vector
        :rtype: np.ndarray
        """
        cos_rot = np.cos(yaw_pitch_roll)
        sin_rot = np.sin(yaw_pitch_roll)

        # pitch
        mat_x = np.array([[1, 0, 0], [0, cos_rot[1], -sin_rot[1]], [0, sin_rot[1], cos_rot[1]]])
        # yaw
        mat_y = np.array([[cos_rot[0], 0, sin_rot[0]], [0, 1, 0], [-sin_rot[0], 0, cos_rot[0]]])
        # roll
        mat_z = np.array([[cos_rot[2], -sin_rot[2], 0], [sin_rot[2], cos_rot[2], 0], [0, 0, 1]])

        return mat_y @ mat_x @ mat_z

    _yaw_pitch_roll: np.ndarray = None
    _mat_rot: np.ndarray

    @classmethod
    def rotate(cls, xyz, yaw_pitch_roll):
        if not np.array_equal(yaw_pitch_roll, cls._yaw_pitch_roll):
            cls._yaw_pitch_roll = yaw_pitch_roll
            cls._mat_rot = cls.rot_matrix(yaw_pitch_roll)
        rotated = np.tensordot(cls._mat_rot, xyz, axes=1)
        return rotated


rotate = MatRot.rotate
