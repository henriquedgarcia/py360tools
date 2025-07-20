from py360tools.transform.transform import ea2xyz, xyz2ea
import numpy as np

from py360tools.assets.projection_base import ProjectionBase


class CMP(ProjectionBase):
    def nm2xyz(self, nm):
        nmface = self.nm2nmface(nm=nm, proj_shape=self.shape)
        vuface = self.nmface2vuface(nmface=nmface, proj_shape=self.shape)
        xyz, face = self.vuface2xyz_face(vuface=vuface)
        return xyz

    def xyz2nm(self, xyz):
        cmp, face = CMP.xyz2nm_face(xyz=xyz, proj_shape=self.shape)
        return cmp

    @staticmethod
    def xyz2nm_face(*, xyz: np.ndarray, proj_shape=None) -> tuple[np.ndarray, np.ndarray]:
        """

        :param proj_shape:
        :param xyz: shape(3, ...)
        :return: nm, face
        """
        vuface = CMP.xyz2vuface(xyz=xyz)
        nmface = CMP.vuface2nmface(vuface=vuface, proj_shape=proj_shape)
        nm, face = CMP.nmface2nm_face(nmface=nmface, proj_shape=proj_shape)
        return nm, face

    @staticmethod
    def nm2ea_face(*, nm: np.ndarray, proj_shape=None) -> tuple[np.ndarray, np.ndarray]:
        nmface = CMP.nm2nmface(nm=nm, proj_shape=proj_shape)
        vuface = CMP.nmface2vuface(nmface=nmface, proj_shape=proj_shape)
        xyz, face = CMP.vuface2xyz_face(vuface=vuface)
        ae = xyz2ea(xyz=xyz)
        return ae, face

    @staticmethod
    def ea2nm_face(*, ea: np.ndarray, proj_shape: tuple = None) -> tuple[np.ndarray, np.ndarray]:
        """
        The face must be a square. proj_shape must have 3:2 ratio
        :param ea: in rad
        :param proj_shape: shape of projection in numpy format: (height, width)
        :return: (nm, face) pixel coord using nearest neighbor
        """
        xyz = ea2xyz(ea=ea)
        nm, face = CMP.xyz2nm_face(xyz=xyz, proj_shape=proj_shape)
        return nm, face

    @staticmethod
    def nm2nmface(nm, proj_shape=None) -> np.ndarray:
        """
        Converts pixel coordinates (n, m) from an image to a 3D cube face coordinate system representation.

        This method processes a set of 2D pixel coordinates (n, m) from an image projection and maps them into
        corresponding 3D cube face coordinates. Each pixel is assigned to one of six cube faces, and its location
        is adjusted relative to that face. The cube faces are indexed as integers from 0 to 5, where the mapping
        takes into account the aspect ratio and dimensions of the cube face in relation to the projection shape.

        :param nm: Pixel coordinates in image projection in the shape of (2, ...), where the first dimension
                   corresponds to the height (n) and width (m) of the image.
        :type nm: np.ndarray
        :param proj_shape: A tuple representing the projection's dimensions. It can either be a 2D shape
                           (height, width) or a 3D shape (N, height, width). If `None`, it defaults to the shape
                           of the provided `nm` array.
        :type proj_shape: np.ndarray or None
        :return: A numpy array in the shape (3, ...), where the third dimension stores the face index and
                 the adjusted (n, m) coordinates relative to the identified face.
        :rtype: np.ndarray
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def xyz2vuface(*, xyz: np.ndarray) -> np.ndarray:
        """

        :param xyz: (3, H, W)
        :return:
        """
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        mx = -x
        my = -y
        mz = -z

        def selection(v1, v2, v3, v4, v5):
            selection1 = np.logical_and(v1, v2)
            selection2 = np.logical_and(selection1, v3)
            selection3 = np.logical_and(selection2, v4)
            selection4 = np.logical_and(selection3, v5)
            return selection4

        face0 = selection(mx >= mz, mx > z, mx >= my, mx > y, x < 0)
        face1 = selection(z >= mx, z > x, z >= my, z > y, z > 0)
        face2 = selection(x >= z, x > mz, x >= my, x > y, x > 0)
        face3 = selection(y >= x, y > mx, y >= mz, y > z, y > 0)
        face4 = selection(mz >= x, mz > mx, mz >= my, mz > y, z < 0)
        face5 = selection(my >= x, my > mx, my >= z, my > mz, y < 0)

        abs_xyz = np.abs(xyz)
        abs_x = abs_xyz[0]
        abs_y = abs_xyz[1]
        abs_z = abs_xyz[2]

        vuface = np.zeros(xyz.shape)

        vuface[0][face0] = y[face0] / abs_x[face0]
        vuface[1][face0] = z[face0] / abs_x[face0]
        vuface[2][face0] = 0

        vuface[0][face1] = y[face1] / abs_z[face1]
        vuface[1][face1] = x[face1] / abs_z[face1]
        vuface[2][face1] = 1

        vuface[0][face2] = y[face2] / abs_x[face2]
        vuface[1][face2] = mz[face2] / abs_x[face2]
        vuface[2][face2] = 2

        vuface[0][face3] = z[face3] / abs_y[face3]
        vuface[1][face3] = mx[face3] / abs_y[face3]
        vuface[2][face3] = 3

        vuface[0][face4] = y[face4] / abs_z[face4]
        vuface[1][face4] = mx[face4] / abs_z[face4]
        vuface[2][face4] = 4

        vuface[0][face5] = mz[face5] / abs_y[face5]
        vuface[1][face5] = mx[face5] / abs_y[face5]
        vuface[2][face5] = 5

        return vuface

    @staticmethod
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

    @staticmethod
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
