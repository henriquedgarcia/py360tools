import numpy as np


class MatRot:
    @staticmethod
    def get_matrix(yaw_pitch_roll):
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

        cos_rot = np.cos(yaw_pitch_roll)
        sin_rot = np.sin(yaw_pitch_roll)

        # pitch
        mat_x = np.array([[1, 0, 0], [0, cos_rot[1], -sin_rot[1]], [0, sin_rot[1], cos_rot[1]]])
        # yaw
        mat_y = np.array([[cos_rot[0], 0, sin_rot[0]], [0, 1, 0], [-sin_rot[0], 0, cos_rot[0]]])
        # roll
        mat_z = np.array([[cos_rot[2], -sin_rot[2], 0], [sin_rot[2], cos_rot[2], 0], [0, 0, 1]])

        return mat_y @ mat_x @ mat_z
