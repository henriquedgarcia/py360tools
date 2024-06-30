import numpy as np

from utils.util import splitx


class Tiling:
    def __init__(self, tiling, proj_res):
        """

        :param tiling: "12x8"
        :type tiling: str
        :param proj_res:
        :type proj_res: str
        """
        self.tiling = tiling
        self.proj_res = proj_res

        self.shape = np.array(splitx(tiling)[::-1])
        self.ntiles = self.shape[0] * self.shape[1]

    def __str__(self):
        return self.tiling

    def __repr__(self):
        return f'Tiling({self})@{self.proj_res}'

    def __eq__(self, other: "Tiling"):
        return (self.tiling == other.tiling and
                tuple(self.proj_res) == tuple(other.proj_res))


class Tile:
    shape: np.ndarray = None
    borders: np.ndarray = None
    borders_xyz: np.ndarray = None
    position_nm: np.ndarray = None

    def __init__(self, tile_id, tiling, image=None):
        """
        # Property
        tiling_position: position of tile in the tiling array
        position: position of tile in the projection image

        :param tile_id: A number on int or str
        :type tile_id: int | str
        :param tiling: Tiling("12x6", projection)
        :type tiling: str
        :param image: array
        """
        self.tile_id = int(tile_id)
        self.tiling = tiling

    def __str__(self):
        return f'tile{self.tile_id}'

    def __repr__(self):
        return f'tile{self.tile_id}@{self.tiling}'

    def __int__(self):
        return self.tile_id

    def __eq__(self, other):
        return (self.tile_id == other.tile_id and
                self.tiling == other.tiling and
                self.shape == other.shape)
