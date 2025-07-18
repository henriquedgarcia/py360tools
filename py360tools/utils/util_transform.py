import cv2
import numpy as np

from py360tools.utils.util import unflatten_index


def check_ea(*, ea):
    _90_deg = np.pi / 2
    _180_deg = np.pi
    _360_deg = 2 * np.pi

    new_ea = np.zeros(ea.shape)
    new_ea[0] = -np.abs(np.abs(ea[0] + _90_deg) - _180_deg) + _90_deg
    new_ea[1] = (ea[1] + _180_deg) % _360_deg - _180_deg

    return new_ea


def get_borders_value(*,
                      array=None,
                      thickness=1
                      ):
    """

    :param array: shape==(C, N, M)
    :type array: list | tuple | np.ndarray | Optional
    :param thickness: How many cells should the borders be thick
    :type thickness: int
    :return: shape==(C, thickness*(2N+2M))
    :rtype: shape==(C, thickness*(2N+2M))
    """
    c = array.shape[0]

    top = array[:, :thickness, :].reshape((c, -1))
    right = array[:, :, :- 1 - thickness:-1].reshape((c, -1))
    left = array[:, :, :thickness].reshape((c, -1))
    bottom = array[:, :- 1 - thickness:-1, :].reshape((c, -1))
    borders_value = np.c_[top, right, bottom, left]
    return np.unique(borders_value, axis=1)


def get_borders_coord_nm(position, shape):
    """
    Calculate the coordinates of the borders of a rectangular area. The function generates
    arrays representing the top, bottom, left, and right borders (in Cartesian coordinates)
    of a rectangle defined by its top-left position and shape dimensions. The resulting
    coordinates are concatenated into a single array.

    :param position: Coordinates of the top-left corner of the rectangle (x, y).
    :type position: list | tuple | np.ndarray
    :param shape: Dimensions of the rectangle (height, width).
    :type shape: list | tuple | np.ndarray
    :return: Concatenated Cartesian coordinates of the boundaries of the rectangle.
             The shape of the result is (2, total numbers of boundary points).
    :rtype: np.ndarray
    """

    x1 = position[1]
    x2 = position[1] + shape[1]
    y1 = position[0]
    y2 = position[0] + shape[0]

    top_border = np.array(np.mgrid[y1:y1 + 1, x1:x2]).reshape(2, -1)
    bottom_border = np.array(np.mgrid[y2 - 1:y2, x1:x2]).reshape(2, -1)
    left_border = np.array(np.mgrid[y1 + 1:y2 - 1, x1:x1 + 1]).reshape(2, -1)
    right_border = np.array(np.mgrid[y1 + 1:y2 - 1, x2 - 1:x2]).reshape(2, -1)

    borders = np.c_[top_border, bottom_border, left_border, right_border]
    return borders


def get_tile_borders(tile_id, tiling_shape, tile_shape):
    """

    :param tile_id: The 1D index on the tiling pattern. (C-style order)
    :type tile_id: int
    :param tiling_shape:
    :type tiling_shape: np.ndarray
    :param tile_shape:
    :type tile_shape: np.ndarray
    :return:
    :rtype: np.ndarray
    """
    tiling_x, tiling_y = unflatten_index(tile_id, tiling_shape)

    x1 = tiling_x * tile_shape[1]
    x2 = (tiling_x + 1) * tile_shape[1]
    y1 = tiling_y * tile_shape[0]
    y2 = (tiling_y + 1) * tile_shape[0]

    top_border = np.array(np.mgrid[y1:y1 + 1, x1:x2]).reshape(2, -1)
    bottom_border = np.array(np.mgrid[y2 - 1:y2, x1:x2]).reshape(2, -1)
    left_border = np.array(np.mgrid[y1 + 1:y2 - 1, x1:x1 + 1]).reshape(2, -1)
    right_border = np.array(np.mgrid[y1 + 1:y2 - 1, x2 - 1:x2]).reshape(2, -1)

    borders = np.c_[top_border, bottom_border, left_border, right_border]
    return borders


def create_nm_coords(shape=(200, 300)):
    nm_test = np.mgrid[0:shape[0], 0:shape[1]]
    return nm_test


def extract_viewport(projection, viewport, proj_img_array):
    """

    :param projection:
    :param viewport:
    :type viewport: Viewport
    :param proj_img_array:
    :type proj_img_array: np.ndarray
    :return:
    :type:
    """

    nm_coord = projection.xyz2nm(viewport.xyz)
    nm_coord = nm_coord.transpose((1, 2, 0))
    vp_img = cv2.remap(proj_img_array,
                       map1=nm_coord[..., 1:2].astype(np.float32),
                       map2=nm_coord[..., 0:1].astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_WRAP)
    # show(vp_img)
    return vp_img
