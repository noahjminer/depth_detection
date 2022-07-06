import cv2
import numpy as np
import sys
import torch
import torchvision

from depth_slicer import DepthSlicer

def precise_pre_process(frame, dims):
    images = []
    for i, dim in enumerate(dims):
        new_slice = frame[dim[2]:dim[3],dim[0]:dim[1]]
        new_slice = cv2.resize(new_slice, (640, 640), interpolation=cv2.INTER_LINEAR)
        images.append(new_slice)
    images.append(cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR))
    return images

def draw_boxes(bboxes, scores, labels, image, draw_dims=False, dims=None, dims_color=(0,255,255)):
    for i, bbox in enumerate(bboxes):
        if labels[i] != 0: continue
        bbox = [int(num) for num in bbox]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,255), 3)
        cv2.putText(image, "{} [{:.2f}]".format(labels[i], float(scores[i])),
                    (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,255), 4)
    if draw_dims: 
        for dim in dims: 
            cv2.rectangle(image, (dim[0], dim[2]), (dim[1], dim[3]), dims_color, 3)
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval().to(device)
model.conf = .2 
model.iou = .25

global_cap = cv2.VideoCapture('D:\\VSProjects]\\DepthRewriteTorch\\camera_8_trim.mp4')

i = 0
while i < 540:
    ret, frame = global_cap.read()
    i += 1


d = DepthSlicer('mask', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), .9, .2, slice_side_length=800, regen_depth=False, regen_dims=False, square_size=50, device=device)
dims = d.dims

images = precise_pre_process(frame, dims)

results = model(images)

bboxes, scores, labels = d.precise_post_process(results)
image = draw_boxes(bboxes, scores, labels, frame, draw_dims=True, dims=d.dims, dims_color=(255,0,255))
cv2.imwrite('frame_isuppress.jpeg', image)

# l t r b
box1 = [30, 50, 50, 70]
box2 = [80, 55, 100, 70]

def box_intersect(box1, box2):
    right = np.minimum(box1[2], box2[2])
    left = np.maximum(box1[0], box2[0])
    top = np.maximum(box1[1], box2[1])
    bottom = np.minimum(box1[3], box2[3])
    return [left, top, right, bottom]

def barea(box):
    return (box[2] - box[0]) * (box[3] - box[1])