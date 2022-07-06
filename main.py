import cv2
import streamlit as st
import torch
import numpy as np
from PIL import Image
from models.common import *

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)


def writebox(result):
    show, save, render, crop, labels = True,False,False,False, True
    crops = []
    for i, (im, pred) in enumerate(zip(result.imgs, result.pred)):
        s = f'image {i + 1}/{len(result.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
        if pred.shape[0]:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                s += f"{n} {result.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            if show or save or render or crop:
                annotator = Annotator(im, example=str(result.names))
                for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                    label = f'{result.names[int(cls)]} {conf:.2f}'
                    if crop:
                        file = save_dir / 'crops' / result.names[int(cls)] / result.files[i] if save else None
                        crops.append({
                            'box': box,
                            'conf': conf,
                            'cls': cls,
                            'label': label,
                            'im': save_one_box(box, im, file=file, save=save)})
                    else:  # all others
                        annotator.box_label(box, label if labels else '', color=colors(cls))
                im = annotator.im
        else:
            s += '(no detections)'

        im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
        return im

path = 'best.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)

while run:
    
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = model(frame)
        
    FRAME_WINDOW.image(writebox(result))
    
else:
    st.write('Stopped')