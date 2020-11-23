import torch
import torchvision
import matplotlib.pyplot as plt
from sutils import *
from tracker import *
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
net = loadmodel()
tracker = PlayerTracker()

frames = cv2.VideoCapture('/home/nam/project/yolov3/data/test.mp4')
fps = frames.get(cv2.CAP_PROP_FPS)
height = int(frames.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(frames.get(cv2.CAP_PROP_FRAME_WIDTH))

fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('../output/topology20_particle250_deque5.mp4', fcc, fps, (width, height))

for i in range(300):
    _, frame = frames.read()
    pred = predict_bbox(net, frame)
    pred = pred.detach().cpu().numpy()
    mask1 = np.logical_and(pred[:, 3] > 500, pred[:, 1] < 670)
    mask2 = np.logical_and(pred[:, 2] > 250, pred[:, 0] < 1750)
    mask = np.logical_and(mask1, mask2)
    #mask = np.logical_and(pred[:, 3] > 450, pred[:, 1] < 700)
    pred = pred[mask, :]

    tracker._tracking(boxes=pred)
    frame = tracker._drawing_boxes(frame)
    #plt.imshow(frame)
    #plt.show()
    out.write(frame)
    print("{0} done".format(i))

frames.release()
out.release()
cv2.destroyAllWindows()

"""
_, frame = frames.read()
pred = predict_bbox(net, frame)

pred = pred.detach().cpu().numpy()
feature_map = torch.from_numpy(frame[:, :, ::-1]/255.0).float()
feature_map = feature_map.permute(2, 0, 1)

tracker.forward(detector_bb=pred, feature_map=feature_map)
tracker.drawing_boxes(frame)

_, frame = frames.read()
pred = predict_bbox(net, frame)
pred = pred.detach().cpu().numpy()
feature_map = torch.from_numpy(frame[:, :, ::-1]/255.0).float()
feature_map = feature_map.permute(2, 0, 1)

tracker.forward(detector_bb=pred, feature_map=feature_map)
tracker.drawing_boxes(frame)
"""
