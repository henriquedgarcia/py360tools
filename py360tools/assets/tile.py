from functools import cached_property
from pathlib import Path
from typing import Union

import numpy as np

from py360tools import ReadVideo
from py360tools.utils.util import get_borders_coord_nm


class Tile:
    def __init__(self, idx: int = None,
                 shape: Union[np.ndarray] = None,
                 position: Union[np.ndarray, tuple[int, int]] = None,
                 path: Path = None,
                 canvas: np.ndarray = None,
                 reader: ReadVideo = None,
                 ):
        self.idx = idx
        self.shape = np.array(shape)
        self.position = np.array(position)
        self.path = path
        self.canvas = canvas
        self.reader = reader

    @cached_property
    def borders(self) -> Union[np.ndarray, tuple[int, int]]:
        borders_coord_nm = get_borders_coord_nm(self.position, self.shape)
        borders_coord_nm = borders_coord_nm.astype(int)
        return borders_coord_nm

    def make_path(self, template: str):
        """

        :param template: uma string que será usada na chamada file_path = template.format(tile=self.idx)
        :return:
        """
        file_path = template.format(tile=self.idx)
        self.path = Path(file_path)
        return self.path

    def __hash__(self):
        return hash(f'{self.idx}@{self.position}@{self.shape}')

    def __str__(self):
        return f'{self.idx}'

    def __repr__(self):
        return f'Tile(id{self.idx}-{self.position}@{self.shape})'

    def __int__(self):
        return self.idx

    def __float__(self):
        return float(self.idx)
