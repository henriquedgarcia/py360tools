from py360tools.models.projectionbase import ProjectionBase
from py360tools.transform.erp_transform import erp2vu, vu2ea, ea2vu, vu2erp
from py360tools.transform.transform import ea2xyz, xyz2ea


class ERP(ProjectionBase):
    def nm2xyz(self, nm):
        vu = erp2vu(nm=nm, proj_shape=self.frame.shape)
        ea = vu2ea(vu=vu)
        xyz = ea2xyz(ea=ea)  # common
        return xyz

    def xyz2nm(self, xyz):
        ea = xyz2ea(xyz=xyz)
        vu = ea2vu(ea=ea)
        nm = vu2erp(vu=vu, proj_shape=self.frame.shape)
        return nm
