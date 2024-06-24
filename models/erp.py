from models.projectionbase import ProjectionBase
from utils.transform import ea2xyz, xyz2ea
from utils.transform_erp import erp_erp2vu, erp_vu2ea, erp_ea2vu, erp_vu2erp


class ERP(ProjectionBase):
    def nm2xyz(self, nm):
        vu = erp_erp2vu(nm=nm, proj_shape=self.shape)
        ea = erp_vu2ea(vu=vu)
        xyz = ea2xyz(ea=ea)  # common
        return xyz

    def xyz2nm(self, xyz):
        ea = xyz2ea(xyz=xyz)
        vu = erp_ea2vu(ea=ea)
        nm = erp_vu2erp(vu=vu, proj_shape=self.shape)
        return nm
