from models.projectionbase import ProjectionBase
from utils.transform_cmp import cmp_cmp2nmface, cmp_nmface2vuface, cmp_vuface2xyz_face, cmp_xyz2vuface, \
    cmp_vuface2nmface, cmp_nmface2cmp_face


class CMP(ProjectionBase):
    def nm2xyz(self, nm):
        nmface = cmp_cmp2nmface(nm=nm, proj_shape=self.shape)
        vuface = cmp_nmface2vuface(nmface=nmface, proj_shape=self.shape)
        xyz, face = cmp_vuface2xyz_face(vuface=vuface)
        return xyz

    def xyz2nm(self, xyz):
        vuface = cmp_xyz2vuface(xyz=xyz)
        nmface = cmp_vuface2nmface(vuface=vuface, proj_shape=self.shape)
        cmp, face = cmp_nmface2cmp_face(nmface=nmface, proj_shape=self.shape)
        return cmp
