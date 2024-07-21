import numpy as np


class WsMse:
    def __init__(self, shape, proj='erp'):
        """

        :param shape:
        :type shape: np.ndarray | tuple
        """
        self.weight_array = {'ERP': None,
                             'CMP': None}

        proj_h, proj_w = shape
        pi_proj = np.pi / proj_h
        proj_h_2 = 0.5 - proj_h / 2
        r = proj_h / 4
        r1 = 0.5 - r
        r2 = r ** 2

        if proj == 'erp':
            def func(y, x):
                w = np.cos((y + proj_h_2) * pi_proj)
                return w
        else:
            def func(y, x):
                x = x % r
                y = y % r
                d = (x + r1) ** 2 + (y + r1) ** 2
                w = (1 + d / r2) ** (-1.5)
                return w

        self.weight_array = np.fromfunction(func, (proj_h, proj_w), dtype=float)

    def compare(self, im_ref, im_deg, tile):

        x1, y1, x2, y2 = tile.position
        weight_tile = self.weight_array[y1:y2, x1:x2]
        wmse = np.sum(weight_tile * (im_ref - im_deg) ** 2) / np.sum(weight_tile)
        return wmse
