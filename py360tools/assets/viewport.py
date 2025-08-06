from functools import cached_property

import cv2
import numpy as np

import py360tools.transform
from py360tools import ProjectionBase
from py360tools.assets.projection_base import Tile
from py360tools.transform.transform import get_vptiles
from py360tools.utils import splitx
from py360tools.utils.util import get_borders_value


class ProjectionError(Exception):
    pass


class Viewport:
    fov: np.ndarray
    vp_shape: np.ndarray

    def __init__(self, resolution, fov, projection: ProjectionBase = None):
        """
        Viewport Class used to extract view pixels in projections.
        The vp is an image as numpy-array with shape (H, M, 3).
        That can be RGB (matplotlib, pillow, etc.) or BGR (opencv).

        :param resolution: '800x600'
        :type resolution: str
        :param fov: '120x90' for (120°x90°)
        :type fov: str
        :param projection: ERP or CMP objects
        :type projection: 'ProjectionBase'
        """
        self.vp_shape = np.array(splitx(resolution)[::-1])
        self.fov = np.deg2rad(np.array(splitx(fov)[::-1]))

        self.set_normals_default()
        self._xyz = self._xyz_default
        self._yaw_pitch_roll = np.array([0., 0., 0.])
        self.projection = projection

    def extract_viewport(self, frame_array, yaw_pitch_roll=None) -> np.ndarray:
        """
        Extracts a viewport image based on the input frame and yaw-pitch-roll orientation.

        This method uses projection data to transform the input frame into a new viewport
        image based on spherical coordinates determined by yaw, pitch, and roll. The extraction
        process involves mapping frame data onto a spherical projection.

        :param frame_array: The input 2D or 3D frame data representing the image or video frame
            to be transformed into a viewport.
        :type frame_array: np.ndarray
        :param yaw_pitch_roll: The yaw, pitch, and roll values representing the orientation for
            viewport transformation, or None for default/default existing orientation.
        :type yaw_pitch_roll: Optional[Tuple[float, float, float]]
        :return: A transformed viewport image mapped according to the specified orientation.
        :rtype: np.ndarray
        :raises ProjectionError: If the projection object is not defined or uninitialized.
        """

        if self.projection is None:
            raise ProjectionError('Projection is not defined.')
        self.yaw_pitch_roll = yaw_pitch_roll if yaw_pitch_roll is not None else self.yaw_pitch_roll
        nm_coord = self.projection.xyz2nm(self.xyz)
        nm_coord = nm_coord.transpose((1, 2, 0))
        vp_img = cv2.remap(frame_array, map1=nm_coord[..., 1:2].astype(np.float32),
                           map2=nm_coord[..., 0:1].astype(np.float32), interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
        # show(vp_img)
        return vp_img

    def get_vptiles(self, yaw_pitch_roll=None) -> list[Tile]:
        """
        Get the tiles used in the viewport.

        This method calculates and returns a list of tiles that are used in the
        viewport based on the specified projection and optionally updates the
        yaw, pitch, and roll of the viewport.

        :param yaw_pitch_roll: Optional yaw, pitch, and roll values to update the
            viewport's orientation.
        :type yaw_pitch_roll: Tuple[float, float, float] or None
        :return: List of tiles showed in the viewport.
        :rtype: list[Tile]
        :raises ProjectionError: If the projection is not defined.
        """

        self.yaw_pitch_roll = self.yaw_pitch_roll if yaw_pitch_roll is None else yaw_pitch_roll
        vptiles = get_vptiles(self.projection, self)
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

    def get_vp_mask(self, lum=255) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param lum: value to draw line
        :return: a numpy.ndarray with one deep color
        """
        canvas = np.zeros(self.projection.shape, dtype='uint8')
        belong = self.is_viewport(self.projection.xyz)
        canvas[belong] = lum
        return canvas

    def get_vp_borders(self, thickness=1, lum=255) -> np.ndarray:
        """
        Draw the borders of the provided viewport using specified thickness and
        luminance. The function projects the sphere using ERP and uses the viewport
        attributes to determine border positions.

        :type thickness: int, optional
        :param lum: Luminance value used for drawing the borders. Defaults to 255.
        :type lum: int, optional
        :return: A numpy array where the specified borders are drawn with the
                 given luminance.
        :rtype: numpy.ndarray
        """
        canvas = np.zeros(self.projection.shape, dtype='uint8')

        vp_borders_xyz = get_borders_value(array=self.xyz, thickness=thickness)
        nm = self.projection.xyz2nm(vp_borders_xyz).astype(int)
        canvas[nm[0, ...], nm[1, ...]] = lum
        return canvas

    def get_vp_tiles(self, lum=255) -> np.ndarray:
        """
        Draw visual perspective (VP) tiles onto a canvas based on the provided viewport.

        This function creates a new canvas matrix initialized to zero, iterates over
        the tiles of the given viewport, and applies a function to draw the borders
        of each tile with the specified luminance value. The result is summed into
        the canvas and returned as a numpy array.

        :param lum: Luminance value for drawing tile borders. Default is 255.
        :type lum: int
        :return: Numpy array representing the canvas with drawn VP tiles.
        :rtype: numpy.ndarray
        """
        canvas = np.zeros(self.projection.shape, dtype='uint8')
        for tile in self.get_vptiles():
            canvas = canvas + self.projection.draw_tile_border(tile, lum=lum)
        return canvas

    @property
    def yaw_pitch_roll(self) -> np.ndarray:
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
        return py360tools.transform.rotate(self._normals, self.yaw_pitch_roll)

    @property
    def xyz(self) -> np.ndarray:
        return py360tools.transform.rotate(self._xyz, self.yaw_pitch_roll)

    _normals: np.ndarray

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
        self._normals = np.array([[0, 0, -cos_fov[1], cos_fov[1]],  # x
                                  [-cos_fov[0], cos_fov[0], 0, 0],  # y
                                  [-sin_fov[0], -sin_fov[0], -sin_fov[1], -sin_fov[1]]  # z
                                  ])

    @cached_property
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

def gnomonic_nm2xyz(nm_coord, fov, vp_shape):
    tan_fov_2 = np.tan(fov / 2)
    y_coord = np.linspace(-tan_fov_2[0], tan_fov_2[0], vp_shape[0], endpoint=True)
    x_coord = np.linspace(-tan_fov_2[1], tan_fov_2[1], vp_shape[1], endpoint=False)

    vp_coord_x, vp_coord_y = np.meshgrid(x_coord, y_coord)
    vp_coord_z = np.ones(vp_shape)
    vp_coord_xyz_ = np.array([vp_coord_x, vp_coord_y, vp_coord_z])

    r = np.sqrt(np.sum(vp_coord_xyz_ ** 2, axis=0, keepdims=True))

    vp_xyz_default: np.ndarray = vp_coord_xyz_ / r  # normalize. final shape==(3,H,W)
    return vp_xyz_default
