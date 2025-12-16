import numpy as np

from py360tools import Projection, splitx, Tile
from py360tools.assets.matrot import MatRot


class SeenTiles:
    _yaw_pitch_roll: np.ndarray
    normals_default: np.ndarray
    normals_rotated: np.ndarray

    def __init__(self, fov: str, projection: Projection):
        """
        :param fov: '120x90' for (120°x90°)
        :type fov: str
        """
        self.projection = projection
        self.fov = np.deg2rad(np.array(splitx(fov)[::-1]))
        self.set_normals_default()
        self.yaw_pitch_roll = np.array([0., 0., 0.])

    def set_normals_default(self):
        """
        Esta descrição está desatualizada. Corrigir.

        Com eixo entrando no observador, rotação horária é negativo e anti-horária
        é positivo. Todos os ângulos são radianos.

        O eixo x aponta para a direita
        O eixo y aponta para baixo
        O eixo z aponta para a frente

        Deslocamento para a direita e para cima é positivo.

        O viewport é a região da esfera que faz intersecção com 4 planos que passam pelo
          centro (4 grandes círculos): cima, direita, baixo e esquerda.
        Os planos são definidos tal que suas normais (N) parte do centro e apontam na mesma direção a
          região do viewport. Ex: O plano de cima aponta para cima, etc.
        Todos os píxels que estiverem abaixo do plano "{N(x, y, z) dot P(x, y, z) ≤ 0}"
        O plano de cima possui inclinação de FOV_Y / 2.
          Sua normal é x=0,y=sin(FOV_Y/2 + pi/2), z=cos(FOV_Y/2 + pi/2)
        O plano de baixo possui inclinação de -FOV_Y / 2.
          Sua normal é x=0,y=sin(-FOV_Y/2 - pi/2), z=cos(-FOV_Y/2 - pi/2)
        O plano da direita possui inclinação de FOV_X / 2. (para direita)
          Sua normal é x=sin(FOV_X/2 + pi/2), y=0, z=cos(FOV_X/2 + pi/2)
        O plano da esquerda possui inclinação de -FOV_X/2. (para direita)
          Sua normal é x=sin(-FOV_X/2 - pi/2), y=0, z=cos(-FOV_X/2 - pi/2)
        :param: fov: the shape of field of view of the viewport
        :type: fov: np.ndarray
        :return: a (3, 4) array with 4 normal vectors in 3D space.
                 For the normal "n", normals_default[:, n] = (x, y, z).
        :rtype: np.ndarray
        """
        fov_2 = self.fov / (2, 2)
        cos_fov = np.cos(fov_2)
        sin_fov = np.sin(fov_2)
        #                       (top, bottom, left, right)
        self.normals_default = np.array([[0, 0, -cos_fov[1], cos_fov[1]],  # x
                                         [-cos_fov[0], cos_fov[0], 0, 0],  # y
                                         [-sin_fov[0], -sin_fov[0], -sin_fov[1], -sin_fov[1]]  # z
                                         ])

    def get_vptiles(self, yaw_pitch_roll) -> list[Tile]:
        """
        Get the tiles used in the viewport.

        This method calculates and returns a list of tiles that are used in the
        viewport based on the specified projection and optionally updates the
        yaw, pitch, and roll of the viewport.

        :param yaw_pitch_roll: Yaw, pitch, and roll values to update the
            viewport's orientation.
        :type yaw_pitch_roll: Tuple[float, float, float]
        :return: List of tiles showed in the viewport.
        :rtype: list[Tile]
        """
        self.yaw_pitch_roll = np.array(yaw_pitch_roll).reshape((3,))
        matrix = MatRot.get_matrix(self.yaw_pitch_roll)
        normals_rotated = np.tensordot(matrix, self.normals_default, axes=1)

        vptiles = []
        for tile in self.projection.tile_list:
            if self.projection.tiling == '1x1':
                vptiles.append(tile)
                continue

            borders_xyz = self.projection.nm2xyz(tile.borders)

            inner_prod = np.tensordot(normals_rotated.T, borders_xyz, axes=1)
            belong = np.all(inner_prod <= 0, axis=0)

            if np.any(belong):
                vptiles.append(tile)
        return vptiles

    def is_viewport(self, x_y_z):
        """
        Verify if the given set of 3D points is within the viewport defined
        by a plane equation in the format of `x1 * m + y1 * n + z1 * z < 0`.
        This method checks whether all points satisfy the plane equation, thereby
        determining if they belong solely to the viewport.

        :param x_y_z: A 3D point array in the shape of (3, ...), where each column
            corresponds to a point in the space [(x, y, z), ...].T.
        :type x_y_z: np.ndarray
        :return: A boolean indicating whether all given points belong to the viewport.
        :rtype: Bool
        """
        inner_prod = np.tensordot(self.normals.T, x_y_z, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        return belong
