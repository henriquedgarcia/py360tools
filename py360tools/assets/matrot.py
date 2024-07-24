import numpy as np


class MatRot:
    _yaw_pitch_roll: np.ndarray
    _matrix: np.ndarray

    def __init__(self):
        self._yaw_pitch_roll = np.array([0., 0., 0.])

    def get_matrix(self, yaw_pitch_roll):
        """
        Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
        See Wikipedia. Use:
            X axis point to right
            Y axis point to down
            Z axis point to front

        Examples
        --------
        >> x, y, z = point
        >> mat = rot_matrix([yaw, pitch, roll])
        >> mat @ (x, y, z)

        :param yaw_pitch_roll: the rotation (yaw, pitch, roll) in rad.
        :type yaw_pitch_roll: np.ndarray | list
        :return: A 3x3 matrix of rotation for (z,y,x) vector
        :rtype: np.ndarray
        """
        if yaw_pitch_roll is not None:
            if np.equal(self._yaw_pitch_roll, yaw_pitch_roll):
                return self._matrix
            self._yaw_pitch_roll = yaw_pitch_roll

        cos_rot = np.cos(self._yaw_pitch_roll)
        sin_rot = np.sin(self._yaw_pitch_roll)

        # pitch
        mat_x = np.array([[1, 0, 0], [0, cos_rot[1], -sin_rot[1]], [0, sin_rot[1], cos_rot[1]]])
        # yaw
        mat_y = np.array([[cos_rot[0], 0, sin_rot[0]], [0, 1, 0], [-sin_rot[0], 0, cos_rot[0]]])
        # roll
        mat_z = np.array([[cos_rot[2], -sin_rot[2], 0], [sin_rot[2], cos_rot[2], 0], [0, 0, 1]])

        self._matrix = mat_y @ mat_x @ mat_z

        return self._matrix


matrot = MatRot()


def rotate(xyz, yaw_pitch_roll):
    matrix = matrot.get_matrix(yaw_pitch_roll)
    rotated = np.tensordot(matrix, xyz, axes=1)
    return rotated

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
