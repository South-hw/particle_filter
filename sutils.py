import torch
import numpy as np
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def loadmodel(img_size=(1088, 1920)):
    cfg = 'cfg/yolov3-spp.cfg'
    weights = 'weights/yolov3-spp-ultralytics.pt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Darknet(cfg, img_size)
    model.load_state_dict((torch.load(weights, map_location=device))['model'])
    model.to(device).eval()

    return model

def predict_bbox(model, frame, img_size=(1088, 1920), conf_thres=0.2, iou_thres=0.3):
    # frame is not tensor and size is (1080, 1920)
    # So, need to resize to img_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    frame_shape = frame.shape
    frame = torch.tensor(frame[None, :,:,::-1]/255.0).to(device=device).float()
    frame = frame.permute(0, 3, 1, 2)
    frame = torch.nn.functional.upsample_bilinear(frame, size=img_size)

    pred = model(frame, augment=False)[0]
    pred = pred.float()
    #pred = non_max_suppression(pred, conf_thres, iou_thres, multi_label=False, classes=None, agnostic=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, multi_label=True)

    pred = pred[0]
    mask = (pred[:, 5] == 0.0)
    pred = pred[mask, :]
    pred[:, :4] = scale_coords(img_size, pred[:, :4], frame_shape).round()
    pred = pred[:, :4]

    return pred

def draw_bb(frame, bbs):
    fig, ax = plt.subplots(1)
    ax.imshow(frame)

    for bb in bbs:
        rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

"""
net = loadmodel()
frames = cv2.VideoCapture('/home/nam/project/yolov3/data/test.mp4')
_, frame = frames.read()
pred = predict_bbox(net, frame)
draw_bb(frame, pred)
"""



