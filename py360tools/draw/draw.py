from typing import Union

import numpy as np
from PIL import Image


def show(array: np.ndarray):
    """
    Displays an image stored as a numpy ndarray using the Pillow library.

    This function takes an array representing an image, converts it to a Pillow
    Image object, and displays it using the built-in Pillow viewer. It does
    not perform any type checking or validation on the input array.

    :param array: A numpy ndarray representing the input image.
    :type array: np.ndarray
    :return: None
    """
    Image.fromarray(array).show()


def array2img(nm_array: np.ndarray,
              shape: tuple = None
              ):
    """
    Displays a 2D representation of the array with nm-coordinates as an image.

    This function generates a binary image where the positions in the input
    array are set to a default value (255) and other positions are left blank.
    It determines the resulting image dimensions based on the `shape` value
    when provided or infers them from the input array's coordinates if `shape`
    is not specified.

          M
       +-->
       |
    N  v

    :param nm_array: 2D numpy array of shape (2, ...), representing the
        coordinates (N, M) used for the placement in the image.
    :param shape: Optional; tuple (N, M) specifying the dimensions of
        the resulting image. If None, the shape will be inferred directly
        from the nm_array.
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
    """
    Composes an image and a mask by overlaying the mask on the image with a given color and opacity.

    This function creates a copy of the provided image and modifies it by blending the areas
    where the mask is active with a specified color. The intensity of the blending is determined by
    the given mask opacity.

    :param img: An image in the form of a NumPy array.
    :param mask: A mask in the form of a NumPy array. Must have the same height and width as `img`.
    :param color: The color used to blend where the mask is active. Defaults to (255,).
    :param mask_opacity: The opacity of the blending effect. Must be a float where 1.0 is fully opaque.
                         Defaults to 1.0.
    :return: A NumPy array representing the image after applying the mask.
    :rtype: np.ndarray
    """
    assert img.shape[:2] == mask.shape[:2], "Image and mask must be the same shape"
    img = img.copy()
    mask_opacity = np.array(mask_opacity)
    img[mask > 0] = (img[mask > 0] * (1 - mask_opacity) + color * mask_opacity).astype('uint8')
    return img


def draw_scatter3d(xyz, vuface):
    """
    Draws a 3D scatter plot using matplotlib based on provided 3D coordinates and face information.
    This function visualizes predefined fixed points and some additional data points from
    given arrays on a 3D plot for better understanding of the spatial distribution.

    :param xyz: A 2D numpy array containing 3D points to be plotted. The array should have 3 rows,
                where each row corresponds to x, y, and z coordinates, respectively.
    :type xyz: numpy.ndarray
    :param vuface: A 2D numpy array containing face-related information to plot categorized data
                   into specific regions. Expected to have 3 rows, where the last row contains
                   categorical face indices.
    :type vuface: numpy.ndarray
    """
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
    """
    Composes images by combining masks, overlays, and resized viewport images.

    The function takes multiple images and applies a series of composites to produce a final
    image. Masks are used to apply color overlays on the projection frame image, and a resized
    viewport image is appended next to it. The function returns the final composed image.

    :param proj_frame_image: The projection frame image used as the base for compositing.
    :type proj_frame_image: Image
    :param all_tiles_borders_image: Mask for all tiles borders in the projection frame image.
    :type all_tiles_borders_image: Image
    :param vp_tiles_image: Mask representing viewport tiles in the projection frame image.
    :type vp_tiles_image: Image
    :param vp_mask_image: Mask defining the viewport area in the projection frame image.
    :type vp_mask_image: Image
    :param vp_borders_image: Mask representing the viewport border overlay.
    :type vp_borders_image: Image
    :param vp_image: The viewport image to be resized and appended to the composition.
    :type vp_image: Image
    :return: Composed image, including modifications to the projection frame and appended
             resized viewport image.
    :rtype: Image
    """
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
