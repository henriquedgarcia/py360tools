from time import time
from typing import Union

import numpy as np
from PIL import Image


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


def show(projection: np.ndarray):
    """
    show image ndarray using Pillow
    :param projection: np.ndarray
    :return:
    """
    Image.fromarray(projection).show()


def array2img(nm_array: np.ndarray,
              shape: tuple = None
              ):
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


def compose_old(proj_frame_image: Image,
                all_tiles_borders_image: Image,
                vp_tiles_image: Image,
                vp_mask_image: Image,
                vp_borders_image: Image,
                vp_image: Image,
                ) -> Image:
    height, width = proj_frame_image.height, proj_frame_image.width

    # Composite mask with projection
    cover_red = Image.new("RGB", (width, height), (255, 0, 0))
    proj_frame_image_c = Image.composite(cover_red, proj_frame_image, mask=all_tiles_borders_image)

    cover_green = Image.new("RGB", (width, height), (0, 255, 0))
    proj_frame_image_c = Image.composite(cover_green, proj_frame_image_c, mask=vp_tiles_image)

    cover_gray = Image.new("RGB", (width, height), (200, 200, 200))
    proj_frame_image_c = Image.composite(cover_gray, proj_frame_image_c, mask=vp_mask_image)

    cover_blue = Image.new("RGB", (width, height), (0, 0, 255))
    proj_frame_image_c = Image.composite(cover_blue, proj_frame_image_c, mask=vp_borders_image)

    # Resize Viewport
    width_vp = int(np.round(height * vp_image.width / vp_image.height))
    vp_image_resized = vp_image.resize((width_vp, height))

    # Compose new image
    new_im = Image.new('RGB', (width + width_vp + 2, height), (255, 255, 255))
    new_im.paste(proj_frame_image_c, (0, 0))
    new_im.paste(vp_image_resized, (width + 2, 0))

    # new_im.show()
    return new_im


def compose(img: np.ndarray,
            mask: np.ndarray,
            color: Union[tuple[int], tuple[int, int, int]] = (255,),
            mask_opacity: float = 1.0
            ):
    assert img.shape[:2] == mask.shape[:2], "Image and mask must be the same shape"

    mask_opacity = np.array(mask_opacity)
    img[mask > 0] = (img[mask > 0] * (1 - mask_opacity) + color * mask_opacity).astype('uint8')
    return img