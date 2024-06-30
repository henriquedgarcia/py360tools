from abc import ABC, abstractmethod

import numpy as np

from utils.lazyproperty import LazyProperty
from transform.transform import rotate


class NormalsInterface(ABC):
    @property
    @abstractmethod
    def normals_default(self):
        """
        Com eixo entrando no observador, rotação horário é negativo e anti-horária
        é positivo. Todos os ângulos são radianos.

        O eixo x aponta para a direita
        O eixo y aponta para baixo
        O eixo z aponta para a frente

        Deslocamento para a direita e para cima é positivo.

        O viewport é a região da esfera que faz intersecção com 4 planos que passam pelo
          centro (4 grandes círculos): cima, direita, baixo e esquerda.
        Os planos são definidos tal que suas normais (N) parte do centro e apontam na mesma direção a
          região do viewport. Ex: O plano de cima aponta para cima, etc.
        Todos os píxels que estiverem abaixo do plano {N(x,y,z) dot P(x,y,z) <= 0}
        O plano de cima possui inclinação de FOV_Y / 2.
          Sua normal é x=0,y=sin(FOV_Y/2 + pi/2), z=cos(FOV_Y/2 + pi/2)
        O plano de baixo possui inclinação de -FOV_Y / 2.
          Sua normal é x=0,y=sin(-FOV_Y/2 - pi/2), z=cos(-FOV_Y/2 - pi/2)
        O plano da direita possui inclinação de FOV_X / 2. (para direita)
          Sua normal é x=sin(FOV_X/2 + pi/2),y=0, z=cos(FOV_X/2 + pi/2)
        O plano da esquerda possui inclinação de -FOV_X/2. (para direita)
          Sua normal é x=sin(-FOV_X/2 - pi/2),y=0, z=cos(-FOV_X/2 - pi/2)

        :return:
        :rtype: None
        """
        pass

    @property
    @abstractmethod
    def normals_rotated(self) -> np.ndarray:
        pass


class VpCoordsInterface(ABC):
    @LazyProperty
    @abstractmethod
    def vp_xyz_default(self):
        """
        The VP projection is based in rectilinear projection.

        In the sphere domain, in te cartesian system, the center of a plain touch the sphere
        on the point (x=0,y=0,z=1).
        The plain sizes are based on the tangent of fov.
        The resolution (number of samples) of viewport is defined by the constructor.
        The proj_coord_xyz.shape is (:,H,W) == (x, y z)
        :return:
        :rtype: None
        """
        pass

    @LazyProperty
    @abstractmethod
    def vp_xyz_rotated(self):
        """

        :return:
        :rtype: np.ndarray
        """
        pass


class ViewportBase(NormalsInterface, VpCoordsInterface, ABC):
    @abstractmethod
    def is_viewport(self, x_y_z):
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x_y_z: A 3D Point list in the space [(x, y, z), ...].T, shape == (3, ...)
        :type x_y_z: np.ndarray
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)
        :rtype: bool

        """
        pass

    @property
    @abstractmethod
    def yaw_pitch_roll(self):
        """

        :return: None
        :return: np.ndarray
        """
        pass

    @yaw_pitch_roll.setter
    @abstractmethod
    def yaw_pitch_roll(self, value):
        """

        :param value:
        :type value:  np.ndarray
        :return: None
        """
        pass


class Viewport(ViewportBase):
    vp_shape: np.ndarray
    fov: np.ndarray

    def __init__(self, vp_shape, fov):
        """
        Viewport Class used to extract view pixels in projections.
        The vp is an image as numpy array with shape (H, M, 3).
        That can be RGB (matplotlib, pillow, etc.) or BGR (opencv).

        :param vp_shape: (600, 800) for 800x600px
        :type vp_shape: np.ndarray
        :param fov: in rad. Ex: "np.array((pi/2, pi*2/3))" for (120°x90°)
        :type fov: np.ndarray
        """
        self.fov = fov
        self.vp_shape = vp_shape

    def is_viewport(self, x_y_z):
        inner_prod = np.tensordot(self.normals_rotated.T, x_y_z, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        return belong

    @LazyProperty
    def normals_default(self):
        """

        :return: a (3, 4) array with 4 normal vectors in 3D space. normals_default[:, n] = (x, y, z) for the normal n.
        """
        fov_2 = self.fov / (2, 2)
        cos_fov = np.cos(fov_2)
        sin_fov = np.sin(fov_2)
        #                           (        top,      bottom,        left,       right)
        normals_default = np.array([[          0,           0, -cos_fov[1],  cos_fov[1]],   # x
                                    [-cos_fov[0],  cos_fov[0],           0,           0],   # y
                                    [-sin_fov[0], -sin_fov[0], -sin_fov[1], -sin_fov[1]]    # z
                                    ])

        return normals_default

    @property
    def normals_rotated(self) -> np.ndarray:
        return rotate(self.normals_default, self.yaw_pitch_roll)

    @LazyProperty
    def vp_xyz_default(self):
        tan_fov_2 = np.tan(self.fov / 2)
        y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True)
        x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False)

        vp_coord_x, vp_coord_y = np.meshgrid(x_coord, y_coord)
        vp_coord_z = np.ones(self.vp_shape)
        vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

        r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))

        vp_xyz_default = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)
        return vp_xyz_default

    @property
    def vp_xyz_rotated(self) -> np.ndarray:
        return rotate(self.vp_xyz_default, self.yaw_pitch_roll)

    _yaw_pitch_roll = np.array([0., 0., 0.])

    @property
    def yaw_pitch_roll(self):
        return self._yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value):
        """

        :param value:
        :type value: np.ndarray | tuple | list | set
        :return:
        """
        self._yaw_pitch_roll = np.array(value).reshape((3,))
