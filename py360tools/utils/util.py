import pickle
from pathlib import Path
from time import time

import numpy as np


def splitx(string: str) -> tuple[int, ...]:
    """
    Receive a string like "5x6x7" (no spaces) and return a tuple of ints, in
    this case, (5, 6, 7).
    :param string: A string of numbers separated with "x".
    :return: Return a list of int
    """
    return tuple(map(int, string.split('x')))


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


def test(func):
    print(f'Testing [{func.__name__}]: ', end='')
    start = time()
    try:
        func()
        print('OK.', end=' ')
    except AssertionError as e:
        print(f'{e.args[0]}', end=' ')
        pass
    final = time() - start
    print(f'Time = {final}')


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


def mse2psnr(_mse: float) -> float:
    return 10 * np.log10((255. ** 2 / _mse))


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


def load_test_data(file_name, default_data):
    """
    if file_name exists, loads it and returns it.
    else, run func(**kwargs), save result in file_name as pickle and return


    :param file_name: The pickle file name.
    :type file_name: Path
    :param default_data:
    :return:
    """
    if file_name.exists():
        return pickle.loads(file_name.read_bytes())

    file_name.parent.mkdir(parents=True, exist_ok=True)
    file_name.write_bytes(pickle.dumps(default_data))
    return default_data


def create_nm_coords(shape=(200, 300)):
    nm_test = np.mgrid[0:shape[0], 0:shape[1]]
    return nm_test


def create_test_default(path, func, *args, **kwargs):
    try:
        data = pickle.loads(path.read_bytes())
    except FileNotFoundError:
        data = func(*args, **kwargs)
        path.write_bytes(pickle.dumps(data))
    return data


def check_ea(*, ea):
    _90_deg = np.pi / 2
    _180_deg = np.pi
    _360_deg = 2 * np.pi

    new_ea = np.zeros(ea.shape)
    new_ea[0] = -np.abs(np.abs(ea[0] + _90_deg) - _180_deg) + _90_deg
    new_ea[1] = (ea[1] + _180_deg) % _360_deg - _180_deg

    return new_ea


def check_deg(axis_name, value):
    """

    :param axis_name:
    :type axis_name: str
    :param value: in rad
    :type value: float
    :return:
    :rtype: float
    """
    n_value = None
    if axis_name == 'azimuth':
        if value >= np.pi or value < -np.pi:
            n_value = (value + np.pi) % (2 * np.pi)
            n_value = n_value - np.pi
        return n_value
    elif axis_name == 'elevation':
        if value > np.pi / 2:
            n_value = 2 * np.pi - value
        elif value < -np.pi / 2:
            n_value = -2 * np.pi - value
        return n_value
    else:
        raise ValueError('"axis_name" not exist.')
