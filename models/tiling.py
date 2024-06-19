import numpy as np

from utils.util import unflatten_index, splitx


class Tiling:
    def __init__(self, tiling, proj_shape, proj_image=None):
        """

        :param tiling: "12x8'
        :type tiling: str
        :param proj_shape: (h: int, w: int)
        :type proj_shape: np.ndarray | tuple
        """
        self.tiling = tiling
        self.proj_shape = proj_shape
        self.proj_image = proj_image

        self.shape = splitx(tiling)[::-1]
        self.ntiles = self.shape[0] * self.shape[1]
        tile_shape = (int(proj_shape[0] / self.shape[0]),
                      int(proj_shape[1] / self.shape[1]))
        self.tiles = [Tile(idx, self, tile_shape) for idx in range(self.ntiles)]

    def __str__(self):
        return self.tiling

    def __repr__(self):
        return f'Tiling({self})@{self.proj_shape}'


class Tile:
    def __init__(self, tile_id, tiling, tile_shape, image=None):
        """
        # Property
        tiling_position: position of tile in the tiling array
        position: position of tile in the projection image

        :param tile_id:
        :type tile_id: int | str
        :param tiling: Tiling("12x6")
        :type tiling: Tiling
        :param tile_shape: (h: int, w: int)
        :type tile_shape: np.ndarray | tuple
        :param image: array
        """
        self.tile_id = int(tile_id)
        self.tiling = tiling
        self.shape = tile_shape
        self.image = image

        self.tiling_position = unflatten_index(tile_id, tiling.shape)
        self.pixel_position = (self.tiling_position[0] * self.shape[1],
                               self.tiling_position[1] * self.shape[0])

    def __str__(self):
        return f'tile{self.tile_id}'

    def __repr__(self):
        return f'tile{self.tile_id}@{self.tiling} ({self.tiling_position[1]}px X {self.tiling_position[0]}px)'

    def __int__(self):
        return self.tile_id
