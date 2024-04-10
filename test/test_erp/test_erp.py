from typing import Union

import numpy as np
from PIL import Image

from erp import ERP
from pathlib import Path
import pickle
from util import test

class TestERP:
    nm_test: np.ndarray
    vu_test: np.ndarray
    xyz_test: np.ndarray
    ea_test: np.ndarray

    def __init__(self):
        self.load_arrays()
        self.test()

    def load_arrays(self):
        self.load_nm_file()
        self.load_vu_file()
        self.load_ea_file()
        self.load_xyz_file()

    def load_nm_file(self):
        nm_file = Path('data_test/ERP_nm.pickle')
        if nm_file.exists():
            self.nm_test = pickle.load(nm_file.open('rb'))
        else:
            shape = (200, 300)
            self.nm_test = np.mgrid[range(shape[0]), range(shape[1])]
            with open(nm_file, 'wb') as f:
                pickle.dump(self.nm_test, f)

    def load_vu_file(self):
        vu_file = Path('data_test/ERP_vu.pickle')
        if vu_file.exists():
            self.vu_test = pickle.load(vu_file.open('rb'))
        else:
            self.vu_test = erp2vu(self.nm_test)
            with open(vu_file, 'wb') as f:
                pickle.dump(self.vu_test, f)

    def load_ea_file(self):
        ea_file = Path('data_test/ERP_ae.pickle')
        if ea_file.exists():
            self.ea_test = pickle.load(ea_file.open('rb'))
        else:
            self.ea_test, face1 = vu2ea(self.vu_test)

            with open(ea_file, 'wb') as f:
                pickle.dump(self.ea_test, f)

    def load_xyz_file(self):
        xyz_file = Path('data_test/ERP_xyz.pickle')
        if xyz_file.exists():
            self.xyz_test = pickle.load(xyz_file.open('rb'))
        else:
            self.xyz_test = ea2xyz(self.ea_test)
            with open(xyz_file, 'wb') as f:
                pickle.dump(self.xyz_test, f)

    def test(self):
        test(self.teste_cmp2mn_face)
        test(self.teste_nmface2vuface)
        test(self.teste_vuface2xyz)
        test(self.teste_cmp2ea)

    def teste_cmp2mn_face(self):
        nmface = cmp2nmface(self.nm_test)
        nm, face = nmface2cmp_face(nmface)

        msg = ''
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in reversion'
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in nmface2cmp_face()'

        nm, face = nmface2cmp_face(self.nmface_test)
        if not np.array_equal(self.nm_test, nm):
            msg += 'Error in cmp2nmface()'

        assert msg == '', msg

    def teste_nmface2vuface(self):
        vuface = nmface2vuface(self.nmface_test)
        nmface = vuface2nmface(vuface)

        msg = ''
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in reversion'
        if not np.array_equal(vuface, self.vu_test):
            msg += 'Error in nmface2vuface()'

        nmface = vuface2nmface(self.vu_test)
        if not np.array_equal(nmface, self.nmface_test):
            msg += 'Error in vuface2nmface()'

        assert msg == '', msg

    def teste_vuface2xyz(self):
        xyz, face = vuface2xyz_face(self.vu_test)
        vuface = xyz2vuface(xyz)

        msg = ''
        if not np.array_equal(vuface, self.vu_test):
            msg += 'Error in reversion'
        if not np.array_equal(xyz, self.xyz_face_test[0]):
            msg += 'Error in vuface2xyz_face()'

        vuface = xyz2vuface(self.xyz_face_test[0])
        if not np.array_equal(vuface, self.vu_test):
            msg += 'Error in xyz2vuface()'

        assert msg == '', msg

    def teste_cmp2ea(self):
        ea, face1 = cmp2ea_face(self.nm_test)
        nm, face2 = ea2cmp_face(ea)

        msg = ''
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in reversion'

        nm, face = ea2cmp_face(self.ea_test)
        if not np.array_equal(ea, self.ea_test):
            msg += 'Error in cmp2ea_face()'
        if not np.array_equal(nm, self.nm_test):
            msg += 'Error in ea2cmp_face()'

        assert msg == '', msg


def compose(proj: ERP, frame_img: Image):
    tiles = proj.get_vptiles()
    frame_array = np.asarray(frame_img)

    height, width = frame_array.shape
    viewport_array = proj.get_viewport(frame_array)
    vp_image = Image.fromarray(viewport_array)
    width_vp = int(np.round(height * vp_image.width / vp_image.height))
    vp_image_resized = vp_image.resize((width_vp, height))

    cover_red = Image.new("RGB", (width, height), (255, 0, 0))
    cover_green = Image.new("RGB", (width, height), (0, 255, 0))
    cover_gray = Image.new("RGB", (width, height), (200, 200, 200))
    cover_blue = Image.new("RGB", (width, height), (0, 0, 255))

    # Get masks
    mask_all_tiles_borders = Image.fromarray(proj.draw_all_tiles_borders())
    mask_vp_tiles = Image.fromarray(proj.draw_vp_tiles())
    mask_vp = Image.fromarray(proj.draw_vp_mask(lum=200))
    mask_vp_borders = Image.fromarray(proj.draw_vp_borders())

    # Composite mask with projection
    frame_img = Image.composite(cover_red, frame_img, mask=mask_all_tiles_borders)
    frame_img = Image.composite(cover_green, frame_img, mask=mask_vp_tiles)
    frame_img = Image.composite(cover_gray, frame_img, mask=mask_vp)
    frame_img = Image.composite(cover_blue, frame_img, mask=mask_vp_borders)

    new_im = Image.new('RGB', (width + width_vp + 2, height), (255, 255, 255))
    new_im.paste(frame_img, (0, 0))
    new_im.paste(vp_image_resized, (width + 2, 0))
    new_im.show()

    print(f'The viewport touch the tiles {tiles}.')


def test_erp():
    # erp '144x72', '288x144','432x216','576x288'
    # cmp '144x96', '288x192','432x288','576x384'
    yaw_pitch_roll = np.deg2rad((70, 0, 0))
    height, width = 288, 576

    ########################################
    # Open Image
    frame_img: Union[Image, list] = Image.open('images/erp1.jpg')
    frame_img = frame_img.resize((width, height))

    erp = ERP(tiling='6x4', proj_res=f'{width}x{height}', fov='100x90')
    erp.yaw_pitch_roll = yaw_pitch_roll
    compose(erp, frame_img)
