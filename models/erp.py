from models.projectionbase import ProjBase
from utils.transform import ea2xyz, xyz2ea
from utils.transform_erp import erp_erp2vu, erp_vu2ea, erp_ea2vu, erp_vu2erp


class ERP(ProjBase):
    def nm2xyz(self, nm, proj_shape=None):
        if proj_shape is None:
            proj_shape = nm.shape[1:]

        vu = erp_erp2vu(nm=nm, proj_shape=proj_shape)
        ea = erp_vu2ea(vu=vu)
        xyz = ea2xyz(ea=ea)  # common
        return xyz

    def xyz2nm(self, xyz, proj_shape=None):
        if proj_shape is None:
            proj_shape = xyz.shape[:2]

        ea = xyz2ea(xyz=xyz)
        vu = erp_ea2vu(ea=ea)
        nm = erp_vu2erp(vu=vu, proj_shape=proj_shape)
        return nm
