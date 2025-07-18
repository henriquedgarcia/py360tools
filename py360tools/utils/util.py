from pathlib import Path
from time import time

import cv2
import numpy as np


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


def make_tile_positions(tiling: str, proj_shape: tuple) -> dict[int, tuple[int, int, int, int]]:
    """
    Calculate tile positions within a given projection resolution and tiling configuration.

    The function calculates the coordinates of each tile in a grid, based on the specified
    tiling resolution and the dimensions of the projection. It ensures that the projection
    resolution is evenly divisible by the tiling resolution, otherwise it raises an error.
    The output is a dictionary where each key corresponds to a unique tile identifier and
    the value is a tuple representing the rectangular area (in terms of start and end
    coordinates) of the tile.

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
    proj_h, proj_w = proj_shape

    if proj_w % tile_M != 0 or proj_h % tile_N != 0:
        raise ValueError(f'The projection resolution ({proj_shape=}) must be a multiple of the tile resolution ({tiling=}).')

    tile_w, tile_h = proj_w // tile_M, proj_h // tile_N

    tile_positions = {}
    for tile in range(tile_N * tile_M):
        tile_m, tile_n = np.unravel_index(tile, (tile_N, tile_M))
        tile_y = tile_h * int(tile_n)
        tile_x = tile_w * int(tile_m)
        x_ini = tile_x
        x_end = tile_x + tile_w
        y_ini = tile_y
        y_end = tile_y + tile_h
        tile_positions[tile] = (x_ini, x_end, y_ini, y_end)
    return tile_positions


def iter_video(video_path: Path, gray=True, dtype='float64'):
    """Iterate over frames in a video file.

    Args:
        video_path: Path to the video file
        gray: If True, convert frames to grayscale
        dtype: NumPy dtype for frame arrays

    Yields:
        np.ndarray: Video frames as NumPy arrays

    Raises:
        ValueError: If the video file cannot be opened
    """
    cap = cv2.VideoCapture(f'{video_path}')
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame.astype(dtype)
    finally:
        cap.release()
