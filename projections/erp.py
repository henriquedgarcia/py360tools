from typing import Union

import numpy as np

from projections.projectionbase import ProjBase
from utils.transform import ea2xyz, xyz2ea
from utils.transform import erp_erp2vu, erp_vu2ea, erp_ea2vu, erp_vu2erp


# from PIL import Image


# from .util import compose


class ERP(ProjBase):
    def nm2xyz(self,
               nm: np.ndarray,
               proj_shape: Union[np.ndarray, tuple]
               ) -> np.ndarray:
        if proj_shape is None:
            proj_shape = nm.shape[1:]

        vu = erp_erp2vu(nm=nm,
                        proj_shape=proj_shape)
        ea = erp_vu2ea(vu=vu)
        xyz = ea2xyz(ea=ea)  # common
        return xyz

    def xyz2nm(self,
               xyz: np.ndarray,
               proj_shape: np.ndarray = None
               ) -> np.ndarray:
        """
        ERP specific.

        :param xyz: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param proj_shape: the shape of projection that cover all sphere. tuple as (N, M)
        :return:
        """
        if proj_shape is None:
            proj_shape = xyz.shape[:2]

        ea = xyz2ea(xyz=xyz)
        vu = erp_ea2vu(ea=ea)
        nm = erp_vu2erp(vu=vu,
                        proj_shape=proj_shape)

        return nm
