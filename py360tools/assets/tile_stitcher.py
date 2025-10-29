import numpy as np
from PIL import Image

from py360tools import Tile, ProjectionBase
from py360tools.assets.read_video import ReadVideo


class TileStitcher:
    """A class to read and stitch video tile chunks into a complete projection frame.

    This class handles reading multiple video tiles and combining them into a single
    projection frame based on specified tiling configuration and resolution.

    Attributes:
        tiles_reader (dict): Dictionary of video tile iterators
        tile_positions (dict): Dictionary of tile position coordinates 
        canvas (np.ndarray): The projection frame buffer

    """
    tiles_reader: dict[Tile, ReadVideo] = None
    canvas: np.ndarray = None

    def __init__(self,
                 tiles_seen: list[Tile],
                 proj_obj: ProjectionBase,
                 gray=True
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
        # todo: No futuro quero fazer com que o TileStitcher receba apenas um objeto projection
        self.tiles_seen = tiles_seen  # converter para lista de int
        self.proj_obj = proj_obj
        self.gray = gray

    @property
    def proj_shape(self):
        return self.proj_obj.shape

    def __iter__(self):
        """
        Returns the iterator object itself.
        Initializes (or re-initializes) tile readers and the canvas for a new iteration.
        """
        self.tiles_reader = {tile: iter(ReadVideo(tile.path, gray=self.gray, dtype='float64'))
                             for tile in self.tiles_seen}
        return self

    def __next__(self) -> np.ndarray:
        """
        Composes and returns the next projected frame by stitching frames from all tiles.

        This method is called by the built-in `next()` function or during iteration
        (e.g., in a `for` loop). It clears the canvas, retrieves the next frame from
        each tile's video stream, stitches them onto the canvas, and returns the
        composed frame.

        OBS. For each iteration the canvas is updated with the next frame from each tile.
        So, if you want to use an anterior frame ex: diference(old_frame, new_frame),
        you need to create a copy

        :return: A numpy array representing the composed frame.
        :rtype: numpy.ndarray
        :raises StopIteration: When all tile streams are exhausted and no more frames
            can be composed.
        :raises Exception: For general errors during the frame stream (maybe OpenCV or FFmpeg).
        """
        self.canvas = np.zeros(self.proj_obj.shape, dtype='uint8')
        if self.tiles_reader is None or self.canvas is None:
            raise StopIteration("Iterator not initialized or already exhausted.")
        return self.mount_canvas()

    def mount_canvas(self):
        """
        Constructs the projection frame by iterating through visible tiles, retrieving
        their corresponding frames, and blending them into the canvas. The function
        resets the canvas before overlaying the frames of the tiles currently in view.

        Raises:
            StopIteration: If the tiles_reader has no more frames for a given tile.

        :return: None
        """
        self.canvas = np.zeros(self.proj_obj.shape, dtype='uint8')
        tile: Tile
        for tile in self.tiles_seen:
            tile_frame = next(self.tiles_reader[tile])
            tile_frame = np.array(Image.fromarray(tile_frame).resize(tile.shape[::-1]))  # às vezes a resolução do vídeo é diferente do objeto projection

            y_ini, x_ini = tile.position
            y_end, x_end = tile.position + tile.shape
            self.canvas[y_ini:y_end, x_ini:x_end] = tile_frame  # caution with projection

        # from PIL import Image
        # Image.fromarray(self.canvas).show()
        return self.canvas

    def reset(self):
        """
        Resets the position of all individual video tile readers to the beginning.
        This allows replaying the stitched video from the start without re-instantiating
        the TileStitcher object.
        """
        if self.tiles_reader:
            for reader in self.tiles_reader.values():
                reader.reset()
