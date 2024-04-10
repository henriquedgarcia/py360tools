from typing import Union, Callable, Optional

import cv2
import numpy as np

from .util import unknown, rot_matrix, get_borders


class ViewportState:
    _mat_rot: Optional[np.ndarray] = unknown
    _rotated_normals: Optional[np.ndarray] = unknown
    _vp_rotated_xyz: Optional[np.ndarray] = unknown
    _vp_borders_xyz: Optional[np.ndarray] = unknown
    _vp_img: Optional[np.ndarray] = unknown
    _is_viewport: Optional[bool] = unknown

    def clean_state(self):
        self._mat_rot: Optional[np.ndarray] = unknown
        self._rotated_normals: Optional[np.ndarray] = unknown
        self._vp_rotated_xyz: Optional[np.ndarray] = unknown
        self._vp_borders_xyz: Optional[np.ndarray] = unknown
        self._vp_img: Optional[np.ndarray] = unknown
        self._is_viewport: Optional[bool] = unknown


class ViewportProps(ViewportState):
    _yaw_pitch_roll: Optional[np.ndarray]
    vp_coord_xyz: np.ndarray
    base_normals: np.ndarray

    @property
    def yaw_pitch_roll(self) -> np.ndarray:
        return self._yaw_pitch_roll

    @yaw_pitch_roll.setter
    def yaw_pitch_roll(self, value: np.ndarray):
        """
        Set a new position to viewport using aerospace's body coordinate system
        and rotate the normals. Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Y-X-Z order. Refer to Wikipedia.

        :param value: the positions like array(yaw, pitch, roll) in rad
        """
        self._yaw_pitch_roll = value
        self.clean_state()

    @property
    def vp_rotated_xyz(self) -> np.ndarray:
        if self._vp_rotated_xyz:
            return self._vp_rotated_xyz

        self._vp_rotated_xyz = np.tensordot(self.mat_rot, self.vp_coord_xyz, axes=1)
        return self._vp_rotated_xyz

    @property
    def mat_rot(self) -> np.ndarray:
        if self._mat_rot:
            return self._mat_rot

        self._mat_rot = rot_matrix(self.yaw_pitch_roll)
        return self._mat_rot

    @property
    def rotated_normals(self) -> np.ndarray:
        if self._rotated_normals:
            return self._rotated_normals

        self._rotated_normals = self.mat_rot @ self.base_normals
        return self._rotated_normals


class Viewport(ViewportProps):
    fov: np.ndarray = unknown
    vp_shape: Union[np.ndarray, tuple] = unknown
    vp_state: set = unknown

    def __init__(self, vp_shape: Union[np.ndarray, tuple], fov: np.ndarray):
        """
        Viewport Class used to extract view pixels in projections.
        The vp is an image as numpy array with shape (H, M, 3).
        That can be RGB (matplotlib, pillow, etc) or BGR (opencv).

        :param frame vp_shape: (600, 800) for 800x600px
        :param fov: in rad. Ex: "np.array((pi/2, pi/2))" for (90°x90°)
        """
        self.fov = fov
        self.vp_shape = vp_shape
        self.vp_state = set()
        self._make_base_normals()
        self._make_base_vp_coord()

        self._yaw_pitch_roll = np.array([0, 0, 0])

    def _make_base_normals(self) -> None:
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
        Todos os pixels que estiverem abaixo do plano {N(x,y,z) dot P(x,y,z) <= 0}
        O plano de cima possui inclinação de FOV_Y / 2.
          Sua normal é x=0,y=sin(FOV_Y/2 + pi/2), z=cos(FOV_Y/2 + pi/2)
        O plano de baixo possui inclinação de -FOV_Y / 2.
          Sua normal é x=0,y=sin(-FOV_Y/2 - pi/2), z=cos(-FOV_Y/2 - pi/2)
        O plano da direita possui inclinação de FOV_X / 2. (para direita)
          Sua normal é x=sin(FOV_X/2 + pi/2),y=0, z=cos(FOV_X/2 + pi/2)
        O plano da esquerda possui inclinação de -FOV_X/2. (para direita)
          Sua normal é x=sin(-FOV_X/2 - pi/2),y=0, z=cos(-FOV_X/2 - pi/2)

        :return:
        """
        fov_y_2, fov_x_2 = self.fov / (2, 2)
        pi_2 = np.pi / 2

        self.base_normals = np.array([[0, -np.sin(fov_y_2 + pi_2), np.cos(fov_y_2 + pi_2)],  # top
                                      [0, -np.sin(-fov_y_2 - pi_2), np.cos(-fov_y_2 - pi_2)],  # bottom
                                      [np.sin(fov_x_2 + pi_2), 0, np.cos(fov_x_2 + pi_2)],  # left
                                      [np.sin(-fov_x_2 - pi_2), 0, np.cos(-fov_x_2 - pi_2)]]).T  # right

    def _make_base_vp_coord(self) -> None:
        """
        The VP projection is based in rectilinear projection.

        In the sphere domain, in te cartesian system, the center of a plain touch the sphere
        on the point (x=0,y=0,z=1).
        The plain sizes are based on the tangent of fov.
        The resolution (number of samples) of viewport is defined by the constructor.
        The proj_coord_xyz.shape is (3,H,W). The dim 0 are x, y z coordinates.
        :return:
        """
        tan_fov_2 = np.tan(self.fov / 2)
        x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], self.vp_shape[1], endpoint=False)
        y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], self.vp_shape[0], endpoint=True)

        (vp_coord_x, vp_coord_y), vp_coord_z = np.meshgrid(x_coord, y_coord), np.ones(self.vp_shape)
        vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

        r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))

        self.vp_coord_xyz = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)

    def is_viewport(self, x_y_z: np.ndarray) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x_y_z: A 3D Point list in the space [(x, y, z), ...].T, shape == (3, ...)
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)

        """
        if self._is_viewport is not None:
            return self._is_viewport

        inner_prod = self.rotated_normals.T @ x_y_z
        px_in_vp = np.all(inner_prod <= 0, axis=0)
        self._is_viewport = np.any(px_in_vp)
        return self._is_viewport

    def get_vp(self, frame: np.ndarray, xyz2nm: Callable) -> np.ndarray:
        """

        :param frame: The projection image.
        :param xyz2nm: A function from 3D to projection.
        :return: The viewport image (RGB)
        """
        if self._vp_img:
            return self._vp_img

        nm_coord: np.ndarray
        nm_coord = xyz2nm(self.vp_rotated_xyz, frame.shape)
        nm_coord = nm_coord.transpose((1, 2, 0))
        self._vp_img = cv2.remap(frame,
                                 map1=nm_coord[..., 1:2].astype(np.float32),
                                 map2=nm_coord[..., 0:1].astype(np.float32),
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP)
        # show2(self._vp_img)
        return self._vp_img

    def get_vp_borders_xyz(self, thickness: int = 1) -> np.ndarray:
        """

        :param thickness: in pixels
        :return: np.ndarray (shape == (1,HxW,3)
        """
        if self._vp_borders_xyz:
            return self._vp_borders_xyz

        self._vp_borders_xyz = get_borders(coord_nm=self.vp_rotated_xyz, thickness=thickness)
        return self._vp_borders_xyz
