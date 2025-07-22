from pathlib import Path

import numpy as np

from py360tools.utils import splitx
from py360tools.utils.util import make_tile_positions, iter_video


class TileStitcher:
    """A class to read and stitch video tile chunks into a complete projection frame.

    This class handles reading multiple video tiles and combining them into a single
    projection frame based on specified tiling configuration and resolution.

    Attributes:
        tiles_reader (dict): Dictionary of video tile iterators
        tile_positions (dict): Dictionary of tile position coordinates 
        canvas (np.ndarray): The projection frame buffer

    """
    tiles_reader: dict
    tile_positions: dict
    canvas: np.ndarray

    def __init__(self,
                 tiles_seen: dict[int, Path],
                 tiling: str,
                 proj_res: str,
                 ):
        """
        Initializes a new instance of the class, setting up the tile mapping and
        projection resolution for further computations.

        :param tiles_seen: A dictionary mapping integer keys to Path objects, which
            represents the collection of tiles observed or processed.
        :param tiling: A string representing the tiling configuration used for
            arranging or processing the tiles.
        :param proj_res: A string indicating the projection resolution, which is
            manipulated to determine the shape of the projected data.
        """
        self.tiles_seen = tiles_seen
        self.proj_shape = splitx(proj_res)[::-1]

        self.tile_positions = make_tile_positions(tiling, self.proj_shape)

    def __iter__(self):
        """
        Iterates through video frames from multiple tiles, composing a projected frame.

        This method initializes an iterator for each video file corresponding to tiles seen
        in `tiles_seen`. It also creates a canvas with the shape specified by `proj_shape`,
        and iteratively generates frames until all tile streams are exhausted. Each frame
        is composed by calling an internal method that mounts the projected frame. If an
        error occurs during iteration, it is caught, logged, and re-raised as appropriate.

        :return: A numpy array representing the composed frame at each iteration.
        :rtype: numpy.ndarray
        :raises StopIteration: When all tile streams are exhausted.
        :raises Exception: For general errors during the iteration process.
        """

        self.tiles_reader = {seen_tile: iter_video(file_path, gray=True)
                             for seen_tile, file_path in self.tiles_seen.items()}
        self.canvas = np.zeros(self.proj_shape, dtype='uint8')

        while True:
            try:
                self._mount_proj_frame()
                yield self.canvas
            except StopIteration:
                break
            except Exception as e:
                print(f"Ocorreu um erro durante a iteração: {e}")
                raise e

    def _mount_proj_frame(self):
        """
        Constructs the projection frame by iterating through visible tiles, retrieving
        their corresponding frames, and blending them into the canvas. The function
        resets the canvas before overlaying the frames of the tiles currently in view.

        Raises:
            StopIteration: If the tiles_reader has no more frames for a given tile.

        :return: None
        """
        self.canvas[:] = 0
        for tile in self.tiles_seen:
            x_ini, x_end, y_ini, y_end = self.tile_positions[tile]
            tile_frame = next(self.tiles_reader[tile])
            self.canvas[y_ini:y_end, x_ini:x_end] = tile_frame
