import streamlit as st
#import joblib
from PIL import Image
#from skimage.transform import resize
import numpy as np
import time
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Designing the interface
st.title("Flower count")
# For newline
st.write('\n')

image = Image.open('AppImage/JADS.png')
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )
# print(str(uploaded_file))

device=''
dnn=False
imgsz = (2112, 2112)
half=False
auto = True
augment=False
visualize=False
conf_thres=0.25
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=1000
line_thickness=3  # bounding box thickness (pixels)



device = select_device(device)
model = DetectMultiBackend('./best_model/best.pt', device=device, dnn=dnn)

stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)

# https://github.com/jalalmansoori19/Cat-Classifier/blob/master/app.py
if uploaded_file is not None:
    # https://github.com/streamlit/streamlit/issues/888
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # show.image(file_bytes, 'Uploaded Image', use_column_width=True)
    img0 = cv2.imdecode(file_bytes, 1)

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride, auto=auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)

    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)

    # print(img.shape)

# For newline
st.sidebar.write('\n')

if st.sidebar.button("Click Here to Count"):

    if uploaded_file is None:
        st.sidebar.write("Please upload an Image to Count")
    else:
        # Half
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        # if pt or jit:
        #     model.model.half() if half else model.model.float()

        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        s = f'image : '
        for i, det in enumerate(pred):
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Print time (inference-only)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if True:#save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None #if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            im0 = annotator.result()

            if False:#view_img:
                cv2.imshow('a', im0)
                cv2.waitKey(0)
                #source = str(source)
            #annt_img = im0.transpose((2, 0, 1))[::-1]
            show.image(im0, 'Uploaded Image', use_column_width=True, channels = 'BGR')
            st.sidebar.header("Algorithm Predicts: ")
            st.sidebar.write(f'{s}Done. ({t3 - t2:.3f}s)')
