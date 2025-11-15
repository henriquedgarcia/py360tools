import numpy as np
from PIL import Image

from py360tools.assets.projection_base import Projection
from py360tools.assets.read_video import ReadVideo
from py360tools.assets.tile import Tile


class TileStitcher:
    """A class to read and stitch video tile chunks into a complete projection frame.

    This class handles reading multiple video tiles and combining them into a single
    projection frame based on specified tiling configuration and resolution.

    Attributes:
        tiles_seen (list[int]): Dictionary of video tile iterators
        tiles_reader (list[Tile]): Dictionary of video tile iterators
        canvas (np.ndarray): The projection frame buffer. Shape==proj_shape

    """
    tiles_reader: list[Tile] = None
    canvas: np.ndarray = None

    def __init__(self, tiles_seen: list[Tile | int], proj_obj: Projection, gray=True):
        """
        Initializes a new instance of the class, setting up the tile mapping and
        projection resolution for further computations.

        :param tiles_seen: A list of Tiles which represents of the tiles observed.
        :param proj_obj: A string indicating the full projection resolution.
        """
        # todo: No futuro quero fazer com que o TileStitcher receba apenas um objeto projection
        self.tiles_seen = list(map(int, tiles_seen))  # converter para lista de int
        self.proj_obj = proj_obj
        self.gray = gray
        self.iterator_init = False
        self.start_canvas()

    @property
    def proj_shape(self):
        return self.proj_obj.shape

    def __iter__(self):
        """
        Returns the iterator object itself.
        Initializes (or re-initializes) tile readers and the canvas for a new 
        iteration.
        """
        self.start_readers()
        self.iterator_init = True
        return self

    def __next__(self) -> np.ndarray:
        """
        Composes and returns the next projected frame by stitching frames from all tiles.

        This method is called by the built-in `next()` function or during iteration
        (e.g., in a `for` loop). It retrieves the next frame from
        each tile's video stream, stitches them onto the canvas, and returns the
        composed frame.

        todo: For each iteration the canvas is updated with the next frame from each tile.
        So, if you want to use an anterior frame ex: diference(old_frame, new_frame),
        you need to create a copy (canvas.copy()):
        ex:
            frame1 = next(tile_stitcher)
            frame2 = next(tile_stitcher)
            frame1 is frame2 is tile_stitcher.canvas
            frame1 - frame2 == 0

            frame1 = next(tile_stitcher)
            frame1 = frame1.copy()
            frame2 = next(tile_stitcher)
            frame1 is not frame2

        :return: A numpy array representing the composed frame.
        :rtype: numpy.ndarray
        :raises StopIteration: When all tile streams are exhausted and no more frames
            can be composed.
        :raises Exception: For general errors during the frame stream (maybe OpenCV or FFmpeg).
        """
        if not self.iterator_init:
            raise StopIteration("Iterator not initialized or exhausted.")

        try:
            self.mount_canvas()
        except StopIteration:
            self.iterator_init = False
            raise

        return self.canvas

    def start_canvas(self):
        if self.gray:
            # shape==(h,w)
            self.canvas = np.zeros((self.proj_shape[0], self.proj_shape[1]), dtype='uint8')
        else:
            # shape==(h,w,3)
            self.canvas = np.zeros((self.proj_shape[0], self.proj_shape[1], 3), dtype='uint8')

    def start_readers(self):
        for tile_int in self.tiles_seen:
            tile_obj: Tile = self.proj_obj.tile_list[int(tile_int)]
            # Update reader
            reader = ReadVideo(tile_obj.path, gray=self.gray, dtype='float64')
            tile_obj.reader = iter(reader)

            # make tile.canvas a view of self.canvas
            y0, x0 = tile_obj.position
            y1, x1 = tile_obj.position + tile_obj.shape
            tile_obj.canvas = self.canvas[y0:y1, x0:x1]

    def mount_canvas(self) -> np.ndarray:
        """
        Constructs the projection frame by iterating through visible tiles, retrieving
        their corresponding frames, and blending them into the canvas. The function
        resets the canvas before overlaying the frames of the tiles currently in view.
        :raises StopIteration: If the tiles_reader has no more frames for a given tile.
        :return: self.canvas
        :rtype: np.ndarray
        """
        # self.canvas[:] = 0

        tile: Tile
        for tile in self.proj_obj.tile_list:
            if tile.reader is None:
                continue

            tile_frame = next(tile.reader)

            # às vezes a resolução do vídeo é diferente do objeto projection
            tile_img = Image.fromarray(tile_frame)
            tile_img = tile_img.resize(tile.shape[::-1])
            tile_frame = np.array(tile_img)

            # lembre de usar [:] para atualizar o canvas, pois tile.canvas é um
            # view de self.canvas.
            tile.canvas[:] = tile_frame

        # Image.fromarray(self.canvas).show()
        return self.canvas

    def reset(self):
        """
        Resets the position of all individual video tile readers to the beginning.
        This allows replaying the stitched video from the start without re-instantiating
        the TileStitcher object.
        """
        for tile in self.proj_obj.tile_list:
            if isinstance(tile.reader, ReadVideo):
                tile.reader.reset()
