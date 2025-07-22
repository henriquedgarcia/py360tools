import cv2
import numpy as np
from PIL import Image
import py360tools as pt

video_capture = cv2.VideoCapture('hog_riders.mp4')
while True:
    ret, frame = video_capture.read()
    if ret:
        frame_h, frame_w, _ = frame.shape

        if frame_w == 0 or frame_h == 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cmp = pt.CMP(proj_res=f'{int(frame_h*3/2)}x{frame_h}', tiling='1x1')
        erp = pt.ERP(proj_res=f'{frame_w}x{frame_h}', tiling='1x1')
        erp_coord = erp.xyz2nm(cmp.xyz)
        
        nm_coord = erp_coord.transpose((1, 2, 0))
        vp_img = cv2.remap(frame, map1=nm_coord[..., 1:2].astype(np.float32),
                           map2=nm_coord[..., 0:1].astype(np.float32), 
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
        img = Image.fromarray(vp_img)
        img.show()


