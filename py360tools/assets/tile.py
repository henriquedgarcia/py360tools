from py360tools.utils.lazyproperty import LazyProperty
from py360tools.utils.util import unflatten_index
from py360tools.utils.util_transform import get_borders_coord_nm


class Tile:
    def __init__(self, tile_id, tiling):
        """
        # Property
        tiling_position: position of tile in the tiling array
        position: position of tile in the projection image

        :param tile_id: A number on int or str
        :type tile_id: int | str
        :param tiling: Tiling("12x6", projection)
        :type tiling: Tiling
        """
        self.tile_id = int(tile_id)
        self.tiling = tiling
        self.tiling_position = unflatten_index(self.tile_id, tiling.shape)  # (x, y)
        self.shape = tiling.tile_shape

        self.position_nm = self.tiling_position * self.shape[::-1]

    @LazyProperty
    def borders_nm(self):
        return get_borders_coord_nm(position=self.position_nm,
                                    shape=self.shape)

    @LazyProperty
    def borders_xyz(self):
        return self.tiling.projection.nm2xyz(self.borders_nm)

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
