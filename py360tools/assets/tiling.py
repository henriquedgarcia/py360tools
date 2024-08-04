import numpy as np

from py360tools.utils.lazyproperty import LazyProperty
from py360tools.utils.util import splitx
from .tile import Tile


class Tiling:
    tiling: str
    proj_res: str
    shape: np.ndarray
    ntiles: int
    tile_shape: np.ndarray

    def __init__(self, tiling, projection):
        """

        :param tiling: "12x8"
        :type tiling: str
        :param projection: 'erp'
        :type projection: ProjectionBase
        """
        self.tiling = tiling
        self.projection = projection

        self.shape = np.array(splitx(tiling)[::-1])
        self.ntiles = self.shape[0] * self.shape[1]
        self.tile_shape = (self.projection.canvas.shape / self.shape).astype(int)

    @LazyProperty
    def tile_list(self):
        return [Tile(tile_id, self) for tile_id in range(self.ntiles)]

    def __str__(self):
        return self.tiling

    def __repr__(self):
        return f'Tiling({self}@{self.projection})'

    def __eq__(self, other: "Tiling"):
        return repr(self) == repr(other)
