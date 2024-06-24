import numpy as np

from models.projectionbase import ProjectionBase
from utils.util import unflatten_index, splitx


class Tiling:
    def __init__(self, tiling, projection):
        """

        :param tiling: "12x8"
        :type tiling: str
        :param projection:
        :type projection: ProjectionBase
        """
        self.tiling = tiling
        self.projection = projection

        self.shape = np.array(splitx(tiling)[::-1])
        self.ntiles = self.shape[0] * self.shape[1]

        self.tile_list: list[Tile] = [Tile(tile_id, self) for tile_id in range(self.ntiles)]

    def __str__(self):
        return self.tiling

    def __repr__(self):
        return f'Tiling({self})@{self.projection}'

    def __eq__(self, other: "Tiling"):
        return (self.tiling == other.tiling and
                tuple(self.projection.proj_res) == tuple(other.projection.proj_res))


class Tile:
    def __init__(self, tile_id, tiling, image=None):
        """
        # Property
        tiling_position: position of tile in the tiling array
        position: position of tile in the projection image

        :param tile_id: A number on int or str
        :type tile_id: int | str
        :param tiling: Tiling("12x6", projection)
        :type tiling: Tiling
        :param image: array
        """
        self.tile_id = int(tile_id)
        self.tiling = tiling
        self.image = image

        self.shape = (tiling.projection.shape / tiling.shape).astype(int)

    _borders: np.ndarray = None

    @property
    def borders(self):
        if self._borders is None:
            tiling_x, tiling_y = unflatten_index(self.tile_id, self.tiling.shape)

            x1 = tiling_x * self.shape[1]
            x2 = (tiling_x + 1) * self.shape[1]
            y1 = tiling_y * self.shape[0]
            y2 = (tiling_y + 1) * self.shape[0]

            top_border = np.mgrid[y1:y1 + 1, x1:x2]
            bottom_border = np.mgrid[y2 - 1:y2, x1:x2]
            left_border = np.mgrid[y1 + 1:y2 - 1, x1:x1 + 1]
            right_border = np.mgrid[y1 + 1:y2 - 1, x2 - 1:x2]

            self._borders = np.c_[top_border, bottom_border, left_border, right_border]
        return self._borders

    @property
    def position(self):
        """

        :return: (x, y) position of tile in the projection image
        """
        return self.borders[::-1, 0]

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
