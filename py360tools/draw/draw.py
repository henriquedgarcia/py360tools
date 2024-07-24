from typing import Union

import numpy as np
from PIL import Image

from py360tools.assets.projection_base import ProjectionBase
from py360tools.utils.util import get_borders_value


def draw_vp_tiles(*, projection: ProjectionBase, lum=255):
    """

    :param projection:
    :param lum:
    :return:
    """
    canvas = np.zeros(projection.frame.shape, dtype='uint8')
    for tile in projection.vptiles:
        canvas = canvas + draw_tile_border(projection=projection, idx=int(tile), lum=lum)
    return canvas


def draw_all_tiles_borders(*, projection: ProjectionBase, lum=255):
    """

    :param projection:
    :param lum:
    :return:
    """
    canvas = np.zeros(projection.frame.shape, dtype='uint8')
    for tile in projection.tiling.tile_list:
        canvas = canvas + draw_tile_border(projection=projection, idx=int(tile), lum=lum)
    return canvas


def draw_tile_border(*, projection: ProjectionBase, idx, lum=255) -> np.ndarray:
    """

    :param projection:
    :param idx:
    :type idx: int
    :param lum:
    :return:
    """
    canvas = np.zeros(projection.frame.shape, dtype='uint8')
    canvas[projection.tiling.tile_list[idx].borders_nm[0], projection.tiling.tile_list[idx].borders_nm[1]] = lum
    return canvas


def draw_vp_mask(*, projection: ProjectionBase, lum=255) -> np.ndarray:
    """
    Project the sphere using ERP. Where is Viewport the
    :param projection:
    :param lum: value to draw line
    :return: a numpy.ndarray with one deep color
    """
    canvas = np.zeros(projection.frame.shape, dtype='uint8')
    belong = projection.viewport.is_viewport(projection.xyz)
    canvas[belong] = lum

    return canvas


def draw_vp_borders(*, projection: ProjectionBase, thickness=1, lum=255):
    """
    Project the sphere using ERP. Where is Viewport the
    :param projection:
    :param lum: value to draw line
    :param thickness: in pixel.
    :return: a numpy.ndarray with one deep color
    """
    canvas = np.zeros(projection.frame.shape, dtype='uint8')

    vp_borders_xyz = get_borders_value(array=projection.viewport.xyz, thickness=thickness)
    nm = projection.xyz2nm(vp_borders_xyz).astype(int)
    canvas[nm[0, ...], nm[1, ...]] = lum
    return canvas


def show(array: np.ndarray):
    """
    show image ndarray using Pillow
    :param array: np.ndarray
    :return:
    """
    Image.fromarray(array).show()


def array2img(nm_array: np.ndarray,
              shape: tuple = None
              ):
    """
          M
       +-->
       |
    N  v

    Show the array with nm coordinates in an image.
    :param nm_array: shape (2, ...)
    :param shape: tuple (N, M)
    :return: None
    """
    if shape is None:
        shape = nm_array.shape[1:]
        if len(shape) < 2:
            shape = (np.max(nm_array[0]) + 1, np.max(nm_array[1]) + 1)
    array2 = np.zeros(shape, dtype=int)[nm_array[0], nm_array[1]]
    array2[:, :] = 255
    show(array2)


def compose(img: np.ndarray,
            mask: np.ndarray,
            color: Union[tuple[int], tuple[int, int, int]] = (255,),
            mask_opacity: float = 1.0
            ):
    assert img.shape[:2] == mask.shape[:2], "Image and mask must be the same shape"

    mask_opacity = np.array(mask_opacity)
    img[mask > 0] = (img[mask > 0] * (1 - mask_opacity) + color * mask_opacity).astype('uint8')
    return img


def draw_scatter3d(xyz, vuface):
    import matplotlib.pyplot as plt
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')

    ax.scatter(0, 0, 0, marker='o', color='red')
    ax.scatter(1, 1, 1, marker='o', color='red')
    ax.scatter(1, 1, -1, marker='o', color='red')
    ax.scatter(1, -1, 1, marker='o', color='red')
    ax.scatter(1, -1, -1, marker='o', color='red')
    ax.scatter(-1, 1, 1, marker='o', color='red')
    ax.scatter(-1, 1, -1, marker='o', color='red')
    ax.scatter(-1, -1, 1, marker='o', color='red')
    ax.scatter(-1, -1, -1, marker='o', color='red')
    [ax.scatter(x, y, z, marker='o', color='red')
     for x, y, z in zip(xyz[0, 0:4140:100],
                        xyz[1, 0:4140:100],
                        xyz[2, 0:4140:100])]

    face0 = vuface[2] == 0
    face1 = vuface[2] == 1
    face2 = vuface[2] == 2
    face3 = vuface[2] == 3
    face4 = vuface[2] == 4
    face5 = vuface[2] == 5
    [ax.scatter(-1, v, u, marker='o', color='blue')
     for v, u in zip(vuface[0, face0][::25], vuface[1, face0][::25])]
    [ax.scatter(u, v, 1, marker='o', color='blue')
     for v, u in zip(vuface[0, face1][::25], vuface[1, face1][::25])]
    [ax.scatter(1, v, -u, marker='o', color='blue')
     for v, u in zip(vuface[0, face2][::25], vuface[1, face2][::25])]
    [ax.scatter(-u, 1, v, marker='o', color='blue')
     for v, u in zip(vuface[0, face3][::25], vuface[1, face3][::25])]
    [ax.scatter(-u, v, -1, marker='o', color='blue')
     for v, u in zip(vuface[0, face4][::25], vuface[1, face4][::25])]
    [ax.scatter(-u, -1, 1, marker='o', color='blue')
     for v, u in zip(vuface[0, face5][::25], vuface[1, face5][::25])]
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


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
