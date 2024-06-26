from models.projectionbase import ProjectionBase
from transform.cmp_transform import cmp2nmface, nmface2vuface, vuface2xyz_face, xyz2vuface, vuface2nmface, \
    nmface2cmp_face


class CMP(ProjectionBase):
    def nm2xyz(self, nm):
        nmface = cmp2nmface(nm=nm, proj_shape=self.shape)
        vuface = nmface2vuface(nmface=nmface, proj_shape=self.shape)
        xyz, face = vuface2xyz_face(vuface=vuface)
        return xyz

    def xyz2nm(self, xyz):
        vuface = xyz2vuface(xyz=xyz)
        nmface = vuface2nmface(vuface=vuface, proj_shape=self.shape)
        cmp, face = nmface2cmp_face(nmface=nmface, proj_shape=self.shape)
        return cmp
