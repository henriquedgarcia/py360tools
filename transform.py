from PIL import Image
import pickle
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from numpy import linalg


def ______basic_____(): ...


def vp2cart(m, n, proj_shape, fov_shape):
    """
    Viewport generation with rectilinear projection

    :param m:
    :param n:
    :param proj_shape: (H, W)
    :param fov_shape: (fov_hor, fov_vert) in degree
    :return:
    """
    proj_h, proj_w = proj_shape
    fov_y, fov_x = map(np.deg2rad, fov_shape)
    half_fov_x, half_fov_y = fov_x / 2, fov_y / 2

    u = (m + 0.5) * 2 * np.tan(half_fov_x) / proj_w
    v = (n + 0.5) * 2 * np.tan(half_fov_y) / proj_h

    x = 1.
    y = -v + np.tan(half_fov_y)
    z = -u + np.tan(half_fov_x)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x = x / r
    y = y / r
    z = z / r

    return x, y, z


def ______utils_____(): ...



def ______test_____(): ...

def show_array(nm_array: np.ndarray, shape: tuple = None):
    """
          M
       +-->
       |
    n  v

    :param nm_array: shape (2, ...)
    :param shape: tuple (N, M)
    :return: None
    """
    if shape is None:
        shape = nm_array.shape[1:]
        if len(shape) < 2:
            shape = (np.max(nm_array[0]) + 1, np.max(nm_array[1]) + 1)
    array2 = np.zeros(shape, dtype=int)[nm_array[0], nm_array[1]] = 255
    Image.fromarray(array2).show()


if __name__ in '__main__':
    # TestCMP()
    TestERP()
