from typing import Union

import numpy as np


def normalize_ea(*,
                 ea
                 ):
    _90_deg = np.pi / 2
    _180_deg = np.pi
    _360_deg = 2 * np.pi

    new_ea = np.zeros(ea.shape)
    new_ea[0] = -np.abs(np.abs(ea[0] + _90_deg) - _180_deg) + _90_deg
    new_ea[1] = (ea[1] + _180_deg) % _360_deg - _180_deg

    return new_ea


def ea2xyz(*,
           ea: np.ndarray
           ) -> np.ndarray:
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
    # xyz_r = np.round(xyz, 6)
    return xyz


def xyz2ea(*, xyz: np.ndarray) -> np.ndarray:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param xyz: shape = (3, ...)
    :return: np.ndarray([azimuth, elevation]) - in rad. shape = (2, ...)
    """
    ea = np.zeros((2,) + xyz.shape[1:])

    r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)

    ea[0] = np.arcsin(-xyz[1] / r)
    ea[1] = np.arctan2(xyz[0], xyz[2])
    ea[1] = (ea[1] + np.pi) % (2 * np.pi) - np.pi

    return ea


def check_deg(axis_name: str, value: float) -> float:
    """

    @param axis_name:
    @param value: in rad
    @return:
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


def rot_matrix(yaw_pitch_roll: Union[np.ndarray, list]) -> np.ndarray:
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
    :return: A 3x3 matrix of rotation for (z,y,x) vector
    """
    cos_rot = np.cos(yaw_pitch_roll)
    sin_rot = np.sin(yaw_pitch_roll)

    # pitch
    mat_x = np.array([[1, 0, 0],
                      [0, cos_rot[1], -sin_rot[1]],
                      [0, sin_rot[1], cos_rot[1]]])
    # yaw
    mat_y = np.array([[cos_rot[0], 0, sin_rot[0]],
                      [0, 1, 0],
                      [-sin_rot[0], 0, cos_rot[0]]])
    # roll
    mat_z = np.array([[cos_rot[2], -sin_rot[2], 0],
                      [sin_rot[2], cos_rot[2], 0],
                      [0, 0, 1]])

    return mat_y @ mat_x @ mat_z


def cmp_cmp2nmface(*,
                   nm: np.ndarray,
                   proj_shape: tuple = None
                   ) -> np.ndarray:
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

    face0 = nmface[2] == 0
    nmface[:2, face0] = nm[:2, face0] % face_size

    face1 = nmface[2] == 1
    nmface[:2, face1] = nm[:2, face1] % face_size

    face2 = nmface[2] == 2
    nmface[:2, face2] = nm[:2, face2] % face_size

    face3 = nmface[2] == 3
    nmface[0][face3] = face_size - nm[1][face3] - 1
    nmface[1][face3] = nm[0][face3] - face_size - 1

    face4 = nmface[2] == 4
    nmface[0][face4] = 2 * face_size - nm[1][face4] - 1
    nmface[1][face4] = nm[0][face4] - face_size - 1

    face5 = nmface[2] == 5
    nmface[0][face5] = 3 * face_size - nm[1][face5] - 1
    nmface[1][face5] = nm[0][face5] - face_size - 1

    return nmface.astype(int)


def cmp_nmface2vuface(*,
                      nmface: np.ndarray,
                      proj_shape=None
                      ) -> np.ndarray:
    """

    :param proj_shape:
    :param nmface: (3, H, W)
    :return:
    """
    vuface = np.zeros(nmface.shape)

    if proj_shape is None:
        proj_shape = nmface.shape

    face_size = proj_shape[-1] / 3
    vuface[:2] = 2 * (nmface[:2, ...] + 0.5) / face_size - 1
    vuface[2] = nmface[2]
    return vuface


def cmp_vuface2xyz_face(*,
                        vuface: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
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


def cmp_xyz2vuface(*,
                   xyz: np.ndarray
                   ) -> np.ndarray:
    """

    :param xyz: (3, H, W)
    :return:
    """

    vuface = np.zeros(xyz.shape)
    abs_xyz = np.abs(xyz)

    def selection(v1,
                  v2,
                  v3,
                  v4,
                  v5
                  ):
        selection1 = np.logical_and(v1,
                                    v2)
        selection2 = np.logical_and(selection1,
                                    v3)
        selection3 = np.logical_and(selection2,
                                    v4)
        selection4 = np.logical_and(selection3,
                                    v5)
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


def cmp_vuface2nmface(*,
                      vuface,
                      proj_shape=None
                      ) -> np.ndarray:
    """

    :param vuface:
    :param proj_shape: (h, w)
    :return:
    """
    nm_face = np.zeros(vuface.shape)
    nm_face[2] = vuface[2]

    if proj_shape is None:
        proj_shape = vuface.shape
    face_size = proj_shape[-1] / 3
    _face_size_2 = face_size / 2
    nm_face[:2] = np.round((vuface[:2, ...] + 1) * _face_size_2 - 0.5)
    return nm_face.astype(int)


def cmp_nmface2cmp_face(*,
                        nmface,
                        proj_shape=None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """

    :param nmface:
    :param proj_shape: (h, w)
    :return:
    """
    new_shape = (2,) + nmface.shape[1:]
    nm = np.zeros(new_shape,
                  dtype=int)

    if proj_shape is None:
        proj_shape = nmface.shape[-2:]
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
    nm[0][face3] = face_size + nmface[1][face3]
    nm[1][face3] = face_size - nmface[0][face3] - 1

    face4 = nmface[2] == 4
    nm[0][face4] = face_size + nmface[1][face4]
    nm[1][face4] = 2 * face_size - nmface[0][face4] - 1

    face5 = nmface[2] == 5
    nm[0][face5] = face_size + nmface[1][face5]
    nm[1][face5] = 3 * face_size - nmface[0][face5] - 1

    face = nmface[2]
    return nm, face


def cmp_xyz2cmp_face(*,
                     xyz: np.ndarray,
                     proj_shape=None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """

    :param proj_shape:
    :param xyz: shape(3, ...)
    :return: nm, face
    """
    vuface = cmp_xyz2vuface(xyz=xyz)
    nmface = cmp_vuface2nmface(vuface=vuface,
                               proj_shape=proj_shape)
    nm, face = cmp_nmface2cmp_face(nmface=nmface,
                                   proj_shape=proj_shape)
    return nm, face


def cmp_cmp2xyz_face(*,
                     nm: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """

    :type nm: np.ndarray
    :return: xyz, face
    """
    nmface = cmp_cmp2nmface(nm=nm)
    vuface = cmp_nmface2vuface(nmface=nmface)
    xyz, face = cmp_vuface2xyz_face(vuface=vuface)
    return xyz, face


def cmp_ea2cmp_face(*,
                    ea: np.ndarray,
                    proj_shape: tuple = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    The face must be a square. proj_shape must have 3:2 ratio
    :param ea: in rad
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (nm, face) pixel coord using nearest neighbor
    """
    if proj_shape is None:
        proj_shape = ea.shape

    xyz = ea2xyz(ea=ea)
    nm, face = cmp_xyz2cmp_face(xyz=xyz,
                                proj_shape=proj_shape)
    return nm, face


def cmp_cmp2ea_face(*,
                    nm: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    xyz, face = cmp_cmp2xyz_face(nm=nm)
    ae = xyz2ea(xyz=xyz)
    return ae, face


def erp_erp2vu(*,
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


def erp_vu2ea(*,
              vu: np.ndarray
              ) -> np.ndarray:
    shape = [2]
    for i in range(len(vu.shape) - 1):
        shape.append(1)

    n1 = np.asarray([-np.pi, 2 * np.pi]).reshape(shape)
    n2 = np.asarray([np.pi / 2, -np.pi]).reshape(shape)

    ea = (vu * n1) + n2
    return ea


def erp_ea2vu(*,
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


def erp_vu2erp(*,
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


def erp_ea2erp(*,
               ea: np.ndarray,
               proj_shape=None
               ) -> np.ndarray:
    """

    :param ea: in rad
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (m, n) pixel coord using nearest neighbor
    """
    ea = normalize_ea(ea=ea)
    vu = erp_ea2vu(ea=ea)
    nm = erp_vu2erp(vu=vu,
                    proj_shape=proj_shape)
    return nm


def erp_erp2ea(*,
               nm: np.ndarray,
               proj_shape=None
               ) -> np.ndarray:
    vu = erp_erp2vu(nm=nm,
                    proj_shape=proj_shape)
    ea = erp_vu2ea(vu=vu)
    return ea
