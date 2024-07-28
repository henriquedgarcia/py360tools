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
