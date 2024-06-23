from models.projectionbase import ProjBase
from utils.transform_cmp import cmp_cmp2nmface, cmp_nmface2vuface, cmp_vuface2xyz_face, cmp_xyz2vuface, \
    cmp_vuface2nmface, cmp_nmface2cmp_face


class CMP(ProjBase):
    def nm2xyz(self, nm, proj_shape=None):
        if proj_shape is None:
            proj_shape = nm.shape[:2]

        nmface = cmp_cmp2nmface(nm=nm, proj_shape=proj_shape)
        vuface = cmp_nmface2vuface(nmface=nmface, proj_shape=proj_shape)
        xyz, face = cmp_vuface2xyz_face(vuface=vuface)
        return xyz

    def xyz2nm(self, xyz, proj_shape=None):
        if proj_shape is None:
            proj_shape = xyz.shape[:2]

        vuface = cmp_xyz2vuface(xyz=xyz)
        nmface = cmp_vuface2nmface(vuface=vuface, proj_shape=proj_shape)
        cmp, face = cmp_nmface2cmp_face(nmface=nmface, proj_shape=proj_shape)
        return cmp
