import numpy as np

from py360tools.transform.transform import ea2xyz, xyz2ea


def nm2nmface(*, nm, proj_shape=None) -> np.ndarray:
    """

    :param nm: shape(2, ...)
               pixel coords in image; n = height, m = width
    :type nm: np.ndarray
    :param proj_shape: (high, width) or (N, high, width)
    :type proj_shape: np.ndarray
    :return: nm_face(3, ...) --> nm_face[:, 0, 0] = (n, m, face)
    """
    new_shape = (3,) + nm.shape[1:]
    nmface = np.zeros(new_shape)

    if proj_shape is None:
        proj_shape = nm.shape

    face_size = proj_shape[-1] // 3  # The face is always a square. todo: check 3:2 aspect ratio
    nmface[2] = nm[1] // face_size + (nm[0] // face_size) * 3

    face0 = nmface[2] == 0
    nmface[:2, face0] = nm[:2, face0] % face_size

    face1 = nmface[2] == 1
    nmface[:2, face1] = nm[:2, face1] % face_size

    face2 = nmface[2] == 2
    nmface[:2, face2] = nm[:2, face2] % face_size

    face3 = nmface[2] == 3
    nmface[0][face3] = face_size - nm[1][face3] % face_size - 1
    nmface[1][face3] = nm[0][face3] % face_size

    face4 = nmface[2] == 4
    nmface[0][face4] = face_size - nm[1][face4] % face_size - 1
    nmface[1][face4] = nm[0][face4] % face_size

    face5 = nmface[2] == 5
    nmface[0][face5] = face_size - nm[1][face5] % face_size - 1
    nmface[1][face5] = nm[0][face5] % face_size

    return nmface.astype(int)


def nmface2vuface(*, nmface: np.ndarray, proj_shape=None) -> np.ndarray:
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


def vuface2xyz_face(*, vuface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def xyz2vuface(*, xyz: np.ndarray) -> np.ndarray:
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

    face0 = selection(-xyz[0] >= -xyz[2], -xyz[0] > xyz[2], -xyz[0] >= -xyz[1], -xyz[0] > xyz[1], xyz[0] < 0)
    vuface[2][face0] = 0
    vuface[1][face0] = xyz[2][face0] / abs_xyz[0][face0]
    vuface[0][face0] = xyz[1][face0] / abs_xyz[0][face0]

    face1 = selection(xyz[2] >= -xyz[0], xyz[2] > xyz[0], xyz[2] >= -xyz[1], xyz[2] > xyz[1], xyz[2] > 0)
    vuface[2][face1] = 1
    vuface[1][face1] = xyz[0][face1] / abs_xyz[2][face1]
    vuface[0][face1] = xyz[1][face1] / abs_xyz[2][face1]

    face2 = selection(xyz[0] >= xyz[2], xyz[0] > -xyz[2], xyz[0] >= -xyz[1], xyz[0] > xyz[1], xyz[0] > 0)
    vuface[2][face2] = 2
    vuface[1][face2] = -xyz[2][face2] / abs_xyz[0][face2]
    vuface[0][face2] = xyz[1][face2] / abs_xyz[0][face2]

    face3 = selection(xyz[1] >= xyz[0], xyz[1] > -xyz[0], xyz[1] >= -xyz[2], xyz[1] > xyz[2], xyz[1] > 0)
    vuface[2][face3] = 3
    vuface[1][face3] = -xyz[0][face3] / abs_xyz[1][face3]
    vuface[0][face3] = xyz[2][face3] / abs_xyz[1][face3]

    face4 = selection(-xyz[2] >= xyz[0], -xyz[2] > -xyz[0], -xyz[2] >= -xyz[1], -xyz[2] > xyz[1], xyz[2] < 0)
    vuface[2][face4] = 4
    vuface[1][face4] = -xyz[0][face4] / abs_xyz[2][face4]
    vuface[0][face4] = xyz[1][face4] / abs_xyz[2][face4]

    face5 = selection(-xyz[1] >= xyz[0], -xyz[1] > -xyz[0], -xyz[1] >= xyz[2], -xyz[1] > -xyz[2], xyz[1] < 0)
    vuface[2][face5] = 5
    vuface[1][face5] = -xyz[0][face5] / abs_xyz[1][face5]
    vuface[0][face5] = -xyz[2][face5] / abs_xyz[1][face5]

    return vuface


def vuface2nmface(*, vuface, proj_shape=None) -> np.ndarray:
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


def nmface2nm_face(*, nmface, proj_shape=None) -> tuple[np.ndarray, np.ndarray]:
    """

    :param nmface:
    :param proj_shape: (h, w)
    :return:
    """
    new_shape = (2,) + nmface.shape[1:]
    nm = np.zeros(new_shape, dtype=int)

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


def xyz2nm_face(*, xyz: np.ndarray, proj_shape=None) -> tuple[np.ndarray, np.ndarray]:
    """

    :param proj_shape:
    :param xyz: shape(3, ...)
    :return: nm, face
    """
    vuface = xyz2vuface(xyz=xyz)
    nmface = vuface2nmface(vuface=vuface, proj_shape=proj_shape)
    nm, face = nmface2nm_face(nmface=nmface, proj_shape=proj_shape)
    return nm, face


def ea2nm_face(*, ea: np.ndarray, proj_shape: tuple = None) -> tuple[np.ndarray, np.ndarray]:
    """
    The face must be a square. proj_shape must have 3:2 ratio
    :param ea: in rad
    :param proj_shape: shape of projection in numpy format: (height, width)
    :return: (nm, face) pixel coord using nearest neighbor
    """
    xyz = ea2xyz(ea=ea)
    nm, face = xyz2nm_face(xyz=xyz, proj_shape=proj_shape)
    return nm, face


def nm2ea_face(*, nm: np.ndarray, proj_shape=None) -> tuple[np.ndarray, np.ndarray]:
    nmface = nm2nmface(nm=nm, proj_shape=proj_shape)
    vuface = nmface2vuface(nmface=nmface, proj_shape=proj_shape)
    xyz, face = vuface2xyz_face(vuface=vuface)
    ae = xyz2ea(xyz=xyz)
    return ae, face
