import numpy as np

from py360tools.assets.matrot import MatRot


class Rotate:
    _yaw_pitch_roll = np.array([0., 0., 0.])
    _rotated: np.ndarray

    @classmethod
    def rotate(cls, xyz, yaw_pitch_roll):
        if not np.equal(cls._yaw_pitch_roll, yaw_pitch_roll):
            cls._yaw_pitch_roll = yaw_pitch_roll
            matrix = MatRot.get_matrix(cls._yaw_pitch_roll)
            cls._rotated = np.tensordot(matrix, xyz, axes=1)
        return cls._rotated
