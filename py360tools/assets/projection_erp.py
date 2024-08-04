from py360tools.assets.projection_base import ProjectionBase
from py360tools.transform.erp_transform import nm2vu, vu2ea, ea2vu, vu2nm
from py360tools.transform.transform import ea2xyz, xyz2ea


class ERP(ProjectionBase):
    def nm2xyz(self, nm):
        vu = nm2vu(nm=nm, proj_shape=self.canvas.shape)
        ea = vu2ea(vu=vu)
        xyz = ea2xyz(ea=ea)  # common
        return xyz

    def xyz2nm(self, xyz):
        ea = xyz2ea(xyz=xyz)
        vu = ea2vu(ea=ea)
        nm = vu2nm(vu=vu, proj_shape=self.canvas.shape)
        return nm
