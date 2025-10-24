from itertools import product
from pathlib import Path
from time import time

import numpy as np

from py360tools.assets.read_video import ReadVideo


def splitx(string: str) -> tuple[int, ...]:
    """
    Receive a string like "5x6x7" (no spaces) and return a tuple of ints, in
    this case, (5, 6, 7).
    :param string: A string of numbers separated with "x".
    :return: Return a list of int
    """
    return tuple(map(int, string.split('x')))


def test(func):
    print(f'Testing [{func.__name__}]: ', end='')
    start = time()
    try:
        func()
        print('OK.', end=' ')
    except AssertionError as e:
        print(f'{e.args[0]}', end=' ')
        pass
    print(f'Time = {time() - start}')


def unflatten_index(idx, shape):
    """

    :param idx: flat index of shape
    :type idx: int
    :param shape: (height, width)
    :type shape: tuple | np.ndarray
    :return: position = (pos_x, pos_y)
    :rtype: tuple[int, int]
    """
    pos_x = idx % shape[1]
    pos_y = idx // shape[1]
    position = (pos_x, pos_y)
    return position


def flatten_index(position, shape):
    """

    :param position: position = (pos_x, pos_y)
    :type position: tuple[int, int] | np.ndarray
    :param shape: the shape of the array (n_columns, n_rows)
    :type shape: tuple | np.ndarray
    :return:
    """
    n_columns = shape[0]
    flat_index = position[0] + position[1] * n_columns
    return flat_index


def mse2psnr(_mse: float, max_sample=255.) -> float:
    return 10 * np.log10((max_sample ** 2 / _mse))


def make_tile_positions(tiling: str, proj_res: str) -> dict[int, tuple[int, int, int, int]]:
    """
    Calculate tile positions within a given projection resolution and tiling configuration.

    The function calculates the coordinates of each tile in a grid, based on the specified
    tiling resolution and the dimensions of the projection. It ensures that the projection
    resolution is evenly divisible by the tiling resolution, otherwise it raises an error.
    The output is a dictionary where each key corresponds to a unique tile identifier and
    the value is a tuple representing the rectangular area (in terms of start and end
    coordinates) of the tile.

    :param proj_res:
    :param tiling: A tuple where the first element is the number of horizontal tiles
                   and the second element is the number of vertical tiles.
    :param proj_shape: A tuple representing the projection shape, where the first
                       element is the height and the second is the width.
    :return: A dictionary mapping tile indices (int) to their corresponding area
             coordinates. Each coordinate tuple contains four integers representing
             (x start, x end, y start, y end).
    :rtype: Dict[int, tuple[int, int, int, int]]
    :raises ValueError: If the specified projection resolution is not evenly divisible
                        by the tiling resolution.
    """
    tile_M, tile_N = splitx(tiling)
    proj_w, proj_h = splitx(proj_res)

    if proj_w % tile_M != 0 or proj_h % tile_N != 0:
        raise ValueError(f'The projection resolution ({proj_res=}) must be a multiple of the tile resolution ({tiling=}).')

    tile_w, tile_h = proj_w // tile_M, proj_h // tile_N

    tile_positions = {}
    for tile_n in range(tile_N):
        for tile_m in range(tile_M):
            tile_x, tile_y = tile_w * tile_m, tile_h * tile_n
            tile_positions[tile_M * tile_n + tile_m] = (int(tile_x), int(tile_x + tile_w), int(tile_y), int(tile_y + tile_h))
    return tile_positions


def iter_video(video_path: Path, gray=True, dtype='float64'):
    """
    Iterate over frames in a video file.

    The function reads a video file located at the specified path and yields video
    frames as NumPy arrays. If the `gray` parameter is True, the frames will be
    converted to grayscale. The `dtype` parameter specifies the NumPy dtype of the
    frame arrays.

    :param video_path: Path to the video file.
    :type video_path: Path
    :param gray: If True, converts frames to grayscale. Defaults to True.
    :type gray: bool
    :param dtype: NumPy data type for frame arrays. Defaults to 'float64'.
    :type dtype: str
    :return: Yields video frames as NumPy arrays.
    :rtype: np.ndarray
    :raises ValueError: If the video file cannot be opened.
    """
    video = ReadVideo(video_path, gray=gray, dtype=dtype)
    for frame in video:
        yield frame


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

    :param position: Coordinates of the top-left corner of the rectangle (y, x).
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
