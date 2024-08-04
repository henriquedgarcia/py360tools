import numpy as np

from py360tools.utils.util import splitx


class Canvas:
    def __init__(self, resolution):
        """

        :param resolution: A string representing the projection resolution. e.g. '600x3000'
        :type resolution: str
        """

        # build_projection
        self.resolution = resolution
        self.shape = np.array(splitx(self.resolution)[::-1], dtype=int)

    def __str__(self):
        return self.resolution

    def __repr__(self):
        return self.resolution
