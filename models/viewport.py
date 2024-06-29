from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from utils.transform import rot_matrix


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
    @property
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

    @property
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
    def mat_rot(self):
        """

        :return:
        :rtype: np.ndarray
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
        self.yaw_pitch_roll = np.array([0, 0, 0])

    def is_viewport(self, x_y_z):
        inner_prod = np.tensordot(self.normals_rotated.T, x_y_z, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        is_vp = np.any(belong)
        return is_vp

    _default_normals: np.ndarray = None

    @property
    def normals_default(self):
        if self._default_normals is None:
            fov_2 = self.fov / (2, 2)
            cos_fov = np.cos(fov_2)
            sin_fov = np.sin(fov_2)

            self._default_normals = np.array([[0, -cos_fov[0], -sin_fov[0]],  # top
                                              [0, cos_fov[0], -sin_fov[0]],  # bottom
                                              [-cos_fov[1], 0, -sin_fov[1]],  # left
                                              [cos_fov[1], 0, -sin_fov[1]]]).T  # right

        return self._default_normals

    @property
    def normals_rotated(self) -> np.ndarray:
        _normals_rotated = np.tensordot(self.mat_rot, self.normals_default, axes=1)
        return _normals_rotated

    # @property
    # def vp_borders_xyz(self):
    #     vp_borders_xyz = get_borders_value(array=self.vp_xyz_rotated, thickness=thickness)
    #     return vp_borders_xyz

    _default_vp_xyz: np.ndarray = None

    @property
    def vp_xyz_default(self):
        if self._default_vp_xyz is None:
            tan_fov_2 = np.tan(self.fov / 2)
            y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True)
            x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False)

            vp_coord_x, vp_coord_y = np.meshgrid(x_coord, y_coord)
            vp_coord_z = np.ones(self.vp_shape)
            vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

            r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))

            self._default_vp_xyz = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)
        return self._default_vp_xyz

    @property
    def vp_xyz_rotated(self) -> np.ndarray:
        _vp_rotated_xyz = np.tensordot(self.mat_rot, self.vp_xyz_default, axes=1)
        return _vp_rotated_xyz

    _mat_rot: Optional[np.ndarray] = None

    @property
    def mat_rot(self):
        if self._mat_rot is not None:
            return self._mat_rot

        self._mat_rot = rot_matrix(self.yaw_pitch_roll)
        return self._mat_rot

    _yaw_pitch_roll: np.ndarray

    @property
    def yaw_pitch_roll(self):
        return self._yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value):
        if not np.equal(self._yaw_pitch_roll, value):
            self._yaw_pitch_roll = value
            self._mat_rot = None
