from typing import Union

import numpy as np

from projectionbase import ProjBase


class CMP(ProjBase):
    def nm2xyz(self, nm: np.ndarray, shape: tuple = None, return_face=False) -> Union[tuple, np.ndarray]:
        """
        CMP specific.

        :param nm:
        :param shape:
        :param return_face:
        :return:

        """
        if shape is None:
            shape = nm.shape
        nmface = self.cmp2nmface(nm, shape)
        vuface = self.nmface2vuface(nmface, shape)
        xyz, face = self.vuface2xyz_face(vuface)
        if return_face:
            return xyz, face
        return xyz

    def xyz2nm(self, xyz: np.ndarray,
               shape: Union[np.ndarray, tuple] = None,
               round_nm: bool = False,
               return_face=False) -> Union[tuple, np.ndarray]:
        """

        :param xyz: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param shape:
        :param round_nm:
        :param return_face:
        :return:
        """
        """

        :parameter
            xyz: np.ndarray
                [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
            shape: Union[tuple, np.ndarray]
                (M, N)
            round_nm: bool
                Must round pixel position?
            return_face: bool
                Must return face?
            
        :return: Union[tuple, np.ndarray]
            pixel coordinates
        """

        vuface = self.xyz2vuface(xyz)
        nmface = self.vuface2nmface(vuface, proj_shape=shape)
        nm, face = self.nmface2cmp_face(nmface, proj_shape=shape)
        if return_face:
            return nm, face
        return nm

    @staticmethod
    def cmp2nmface(nm: np.ndarray, proj_shape: tuple = None) -> np.ndarray:
        """

        :param proj_shape:
        :param nm: shape(2, ...)
                   pixel coords in image; n = height, m = width
        :return: nm_face(3, ...)
        """
        new_shape = (3,) + nm.shape[1:]
        nmface = np.zeros(new_shape)

        if proj_shape is None:
            proj_shape = nm.shape

        face_size = proj_shape[-1] // 3
        nmface[2] = nm[1] // face_size + (nm[0] // face_size) * 3
        nmface[:2] = nm % face_size

        nmface_rotated = np.rot90(nmface, axes=(2, 1))
        nmface[:, nmface[2] == 3] = nmface_rotated[:, nmface_rotated[2] == 3]
        nmface[:, nmface[2] == 4] = nmface_rotated[:, nmface_rotated[2] == 4]
        nmface[:, nmface[2] == 5] = nmface_rotated[:, nmface_rotated[2] == 5]

        return nmface.astype(int)

    @staticmethod
    def nmface2cmp_face(nmface, proj_shape=None):
        new_shape = (2,) + nmface.shape[1:]
        nm = np.zeros(new_shape, dtype=int)

        if proj_shape is None:
            proj_shape = nmface.shape
        face_size = proj_shape[-1] // 3

        face0 = nmface[2] == 0
        nm[0][face0] = nmface[0][face0]
        nm[1][face0] = nmface[1][face0]

        face1 = nmface[2] == 1
        nm[0][face1] = nmface[0][face1]
        nm[1][face1] = nmface[1][face1] + face_size

        face2 = nmface[2] == 2
        nm[0][face2] = nmface[0][face2]
        nm[1][face2] = nmface[1][face2] + 2 * face_size

        face3 = nmface[2] == 3
        nmface_rotated = np.rot90(nmface, axes=(1, 2))
        nm[0][face3] = nmface_rotated[0][nmface_rotated[2] == 3] + face_size
        nm[1][face3] = nmface_rotated[1][nmface_rotated[2] == 3]

        face4 = nmface[2] == 4
        nm[0][face4] = nmface_rotated[0][nmface_rotated[2] == 4] + face_size
        nm[1][face4] = nmface_rotated[1][nmface_rotated[2] == 4] + face_size

        face5 = nmface[2] == 5
        nm[0][face5] = nmface_rotated[0][nmface_rotated[2] == 5] + face_size
        nm[1][face5] = nmface_rotated[1][nmface_rotated[2] == 5] + 2 * face_size
        face = nmface[2]
        return nm, face

    @staticmethod
    def nmface2vuface(nmface: np.ndarray, proj_shape=None) -> np.ndarray:
        """

        :param proj_shape:
        :param nmface: (3, H, W)
        :return:
        """
        vuface = np.zeros(nmface.shape)

        if proj_shape is None:
            proj_shape = nmface.shape

        face_size = proj_shape[-1] / 3

        normalize = np.vectorize(lambda m: 2 * (m + 0.5) / face_size - 1)
        vuface[:2] = normalize(nmface[:2, ...])
        vuface[2] = nmface[2]
        return vuface

    @staticmethod
    def vuface2nmface(vuface, proj_shape=None):
        nm_face = np.zeros(vuface.shape)
        nm_face[2] = vuface[2]

        if proj_shape is None:
            proj_shape = vuface.shape
        face_size = proj_shape[-1] / 3
        _face_size_2 = face_size / 2

        denormalize = np.vectorize(lambda u: np.round((u + 1) * _face_size_2 - 0.5))
        nm_face[:2] = denormalize(vuface[:2, ...])
        return nm_face.astype(int)

    @staticmethod
    def vuface2xyz_face(vuface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.zeros(vuface.shape)

        face0 = vuface[2] == 0
        xyz[0, face0] = -1
        xyz[1, face0] = vuface[0, face0]
        xyz[2, face0] = vuface[1, face0]

        face1 = vuface[2] == 1
        xyz[0, face1] = vuface[1, face1]
        xyz[1, face1] = vuface[0, face1]
        xyz[2, face1] = 1

        face2 = vuface[2] == 2
        xyz[0, face2] = 1
        xyz[1, face2] = vuface[0, face2]
        xyz[2, face2] = -vuface[1, face2]

        face3 = vuface[2] == 3
        xyz[0, face3] = -vuface[1, face3]
        xyz[1, face3] = 1
        xyz[2, face3] = vuface[0, face3]

        face4 = vuface[2] == 4
        xyz[0, face4] = -vuface[1, face4]
        xyz[1, face4] = vuface[0, face4]
        xyz[2, face4] = -1

        face5 = vuface[2] == 5
        xyz[0, face5] = -vuface[1, face5]
        xyz[1, face5] = -1
        xyz[2, face5] = -vuface[0, face5]
        face = vuface[2]

        return xyz, face

    @staticmethod
    def xyz2vuface(xyz: np.ndarray) -> np.ndarray:
        """

        :param xyz: (3, H, W)
        :return:
        """

        vuface = np.zeros(xyz.shape)
        abs_xyz = np.abs(xyz)

        def selection(v1, v2, v3, v4, v5):
            selection1 = np.logical_and(v1, v2)
            selection2 = np.logical_and(selection1, v3)
            selection3 = np.logical_and(selection2, v4)
            selection4 = np.logical_and(selection3, v5)
            return selection4

        face0 = selection(-xyz[0] >= -xyz[2],
                          -xyz[0] > xyz[2],
                          -xyz[0] >= -xyz[1],
                          -xyz[0] > xyz[1],
                          xyz[0] < 0)
        vuface[2][face0] = 0
        vuface[1][face0] = xyz[2][face0] / abs_xyz[0][face0]
        vuface[0][face0] = xyz[1][face0] / abs_xyz[0][face0]

        face1 = selection(xyz[2] >= -xyz[0],
                          xyz[2] > xyz[0],
                          xyz[2] >= -xyz[1],
                          xyz[2] > xyz[1],
                          xyz[2] > 0)
        vuface[2][face1] = 1
        vuface[1][face1] = xyz[0][face1] / abs_xyz[2][face1]
        vuface[0][face1] = xyz[1][face1] / abs_xyz[2][face1]

        face2 = selection(xyz[0] >= xyz[2],
                          xyz[0] > -xyz[2],
                          xyz[0] >= -xyz[1],
                          xyz[0] > xyz[1],
                          xyz[0] > 0)
        vuface[2][face2] = 2
        vuface[1][face2] = -xyz[2][face2] / abs_xyz[0][face2]
        vuface[0][face2] = xyz[1][face2] / abs_xyz[0][face2]

        face3 = selection(xyz[1] >= xyz[0],
                          xyz[1] > -xyz[0],
                          xyz[1] >= -xyz[2],
                          xyz[1] > xyz[2],
                          xyz[1] > 0)
        vuface[2][face3] = 3
        vuface[1][face3] = -xyz[0][face3] / abs_xyz[1][face3]
        vuface[0][face3] = xyz[2][face3] / abs_xyz[1][face3]

        face4 = selection(-xyz[2] >= xyz[0],
                          -xyz[2] > -xyz[0],
                          -xyz[2] >= -xyz[1],
                          -xyz[2] > xyz[1],
                          xyz[2] < 0)
        vuface[2][face4] = 4
        vuface[1][face4] = -xyz[0][face4] / abs_xyz[2][face4]
        vuface[0][face4] = xyz[1][face4] / abs_xyz[2][face4]

        face5 = selection(-xyz[1] >= xyz[0],
                          -xyz[1] > -xyz[0],
                          -xyz[1] >= xyz[2],
                          -xyz[1] > -xyz[2],
                          xyz[1] < 0)
        vuface[2][face5] = 5
        vuface[1][face5] = -xyz[0][face5] / abs_xyz[1][face5]
        vuface[0][face5] = -xyz[2][face5] / abs_xyz[1][face5]

        return vuface

    def ea2cmp_face(self, ea: np.ndarray, proj_shape: tuple = None) -> tuple[np.ndarray, np.ndarray]:
        """
        The face must be a square. proj_shape must have 3:2 ratio
        :param ea: in rad
        :param proj_shape: shape of projection in numpy format: (height, width)
        :return: (nm, face) pixel coord using nearest neighbor
        """
        if proj_shape is None:
            proj_shape = ea.shape

        xyz = self.ea2xyz(ea)
        nm, face = self.xyz2nm(xyz, shape=proj_shape, return_face=True)
        return nm, face

    def cmp2ea_face(self, nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xyz, face = self.nm2xyz(nm)
        ae = self.xyz2ea(xyz)
        return ae, face
