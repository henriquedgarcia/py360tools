from abc import ABC, abstractmethod

import cv2
import numpy as np

from py360tools.assets.matrot import matrot
from py360tools.utils.lazyproperty import LazyProperty


class ViewportBase(ABC):
    fov: np.ndarray
    vp_shape: np.ndarray

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
        self.vp_shape = vp_shape
        self.fov = fov

    @abstractmethod
    def extract_viewport(self, projection, frame_array):
        """

        :param projection:
        :type projection: "ProjectionBase"
        :param frame_array:
        :type frame_array: np.ndarray
        :return:
        :type:
        """
        pass

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
    def xyz(self):
        """

        :return: None
        :return: np.ndarray
        """
        pass

    @property
    @abstractmethod
    def normals(self):
        """

        :return: None
        :return: np.ndarray
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
    def __init__(self, vp_shape, fov):
        super().__init__(vp_shape, fov)
        self._normals = self._normals_default()
        self._xyz = self._xyz_default
        self._yaw_pitch_roll = np.array([0., 0., 0.])

    def extract_viewport(self, projection, frame_array):
        nm_coord = projection.xyz2nm(self.xyz)
        nm_coord = nm_coord.transpose((1, 2, 0))
        vp_img = cv2.remap(frame_array, map1=nm_coord[..., 1:2].astype(np.float32),
                           map2=nm_coord[..., 0:1].astype(np.float32), interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
        # show(vp_img)
        return vp_img

    def is_viewport(self, x_y_z):
        inner_prod = np.tensordot(self.normals.T, x_y_z, axes=1)
        belong = np.all(inner_prod <= 0, axis=0)
        return belong

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

    @property
    def normals(self) -> np.ndarray:
        return rotate(self._normals, self.yaw_pitch_roll)

    @property
    def xyz(self) -> np.ndarray:
        return rotate(self._xyz, self.yaw_pitch_roll)

    def _normals_default(self):
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
        :param: fov: the shape of field of view of the viewport
        :type: fov: np.ndarray
        :return: a (3, 4) array with 4 normal vectors in 3D space. normals_default[:, n] = (x, y, z) for the normal n.
        :rtype: np.ndarray
        """
        fov_2 = self.fov / (2, 2)
        cos_fov = np.cos(fov_2)
        sin_fov = np.sin(fov_2)
        #        (        top,      bottom,        left,       right)
        _default = np.array([[0, 0, -cos_fov[1], cos_fov[1]],  # x
                             [-cos_fov[0], cos_fov[0], 0, 0],  # y
                             [-sin_fov[0], -sin_fov[0], -sin_fov[1], -sin_fov[1]]  # z
                             ])
        return _default

    @LazyProperty
    def _xyz_default(self):
        tan_fov_2 = np.tan(self.fov / 2)
        y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True)
        x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False)

        vp_coord_x, vp_coord_y = np.meshgrid(x_coord, y_coord)
        vp_coord_z = np.ones(self.vp_shape)
        vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

        r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))

        vp_xyz_default: np.ndarray = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)
        return vp_xyz_default
