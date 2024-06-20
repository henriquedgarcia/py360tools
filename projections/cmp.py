import numpy as np

from projections.projectionbase import ProjBase
from utils.transform import (cmp_cmp2nmface, cmp_nmface2vuface, cmp_vuface2xyz_face, cmp_xyz2vuface, cmp_vuface2nmface,
                             cmp_nmface2cmp_face)


class CMP(ProjBase):
    faces_list = []

    def nm2xyz(self,
               nm,
               proj_shape,
               rotate=True
               ):
        """
        CMP specific.

        :param nm: shape==(2,...)
        :type nm: np.ndarray
        :param proj_shape: (N, M) the shape of projection that cover all sphere
        :type proj_shape: np.ndarray | tuple
        :param rotate: True
        :type rotate: bool
        :return: (x, y, z) in a (3, ...) ndarray
        :rtype: np.ndarray
        """
        nmface = cmp_cmp2nmface(nm=nm,
                                proj_shape=proj_shape)
        vuface = cmp_nmface2vuface(nmface=nmface,
                                   proj_shape=proj_shape)
        xyz, face = cmp_vuface2xyz_face(vuface=vuface)
        return xyz

    def xyz2nm(self,
               xyz: np.ndarray,
               proj_shape = None,
               ):
        """
        CMP specific.

        :param xyz: shape==(3,...)
        :type xyz: np.ndarray
        :param proj_shape: (N, M) the shape of projection that cover all sphere
        :type proj_shape: np.ndarray | tuple
        :return: (x, y, z) in a (3, H, W) ndarray
        :rtype: np.ndarray
        """
        vuface = cmp_xyz2vuface(xyz=xyz)
        nmface = cmp_vuface2nmface(vuface=vuface,
                                   proj_shape=proj_shape)
        cmp, face = cmp_nmface2cmp_face(nmface=nmface,
                                        proj_shape=proj_shape)

        # fig = plt.figure()
        #
        # ax = fig.add_subplot(projection='3d')
        #
        # ax.scatter(0, 0, 0, marker='o', color='red')
        # ax.scatter(1, 1, 1, marker='o', color='red')
        # ax.scatter(1, 1, -1, marker='o', color='red')
        # ax.scatter(1, -1, 1, marker='o', color='red')
        # ax.scatter(1, -1, -1, marker='o', color='red')
        # ax.scatter(-1, 1, 1, marker='o', color='red')
        # ax.scatter(-1, 1, -1, marker='o', color='red')
        # ax.scatter(-1, -1, 1, marker='o', color='red')
        # ax.scatter(-1, -1, -1, marker='o', color='red')
        # [ax.scatter(x, y, z, marker='o', color='red')
        #  for x, y, z in zip(xyz[0, 0:4140:100],
        #                     xyz[1, 0:4140:100],
        #                     xyz[2, 0:4140:100])]
        #
        # face0 = vuface[2] == 0
        # face1 = vuface[2] == 1
        # face2 = vuface[2] == 2
        # face3 = vuface[2] == 3
        # face4 = vuface[2] == 4
        # face5 = vuface[2] == 5
        # [ax.scatter(-1, v, u, marker='o', color='blue')
        #  for v, u in zip(vuface[0, face0][::25], vuface[1, face0][::25])]
        # [ax.scatter(u, v, 1, marker='o', color='blue')
        #  for v, u in zip(vuface[0, face1][::25], vuface[1, face1][::25])]
        # [ax.scatter(1, v, -u, marker='o', color='blue')
        #  for v, u in zip(vuface[0, face2][::25], vuface[1, face2][::25])]
        # [ax.scatter(-u, 1, v, marker='o', color='blue')
        #  for v, u in zip(vuface[0, face3][::25], vuface[1, face3][::25])]
        # [ax.scatter(-u, v, -1, marker='o', color='blue')
        #  for v, u in zip(vuface[0, face4][::25], vuface[1, face4][::25])]
        # [ax.scatter(-u, -1, 1, marker='o', color='blue')
        #  for v, u in zip(vuface[0, face5][::25], vuface[1, face5][::25])]
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        return cmp
