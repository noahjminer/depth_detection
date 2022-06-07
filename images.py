
from depth_slicer import DepthSlicer
from detection_utils import non_max_suppression_fast, draw_boxes

import torch 
import cv2
import time
import numpy as np

def slice_test():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    img_paths = ["D:\\Downloads\\camera_8_trim_Moment.jpg", "D:\\Downloads\\DI3P2L52GBCURMVJEWQTKXOAOY.jpg", "D:\\VSProjects]\\DepthRewriteTorch\\scrapbook_test.jpg" , "D:\\VSProjects]\\DepthRewriteTorch\\1000x1000_test.jpg", "D:\\VSProjects]\\DepthRewriteTorch\\1200x1200.jpg"]
    img = cv2.imread(img_paths[0])
    shape = img.shape

    d = DepthSlicer('precise_grid', img, .90, .2, 1000, True, square_size=50)
    dims = d.dims

    images = []
    for i, dim in enumerate(dims): 
        prev = time.time()
        new_slice = dim.make_grid_image(img)
        cv2.imwrite(f'slice_test_{i}.jpeg', new_slice)
        new_slice = cv2.resize(new_slice, (608, 608), interpolation=cv2.INTER_LINEAR)
        images.append(new_slice)
    images.append(img)

    classes = [0]
    results = model(images)

    final_detections = []
    for index, i in enumerate(results.xyxyn):
        labels, cord_thres = i[:, -1].numpy(), i[:, :-1].numpy()
        if (index < len(dims)):
            for l, label in enumerate(labels): 
                if label == 0: 
                    left, right, top, bottom = dims[index].normal_to_pixel_coords(cord_thres[l])
                    width = right - left 
                    height = bottom - top 
                    final_detections.append((int(label), cord_thres[l][4], (left,top,width,height)))
        else: 
            for l, label in enumerate(labels): 
                if label == 0: 
                    left = cord_thres[l][0] * shape[1]
                    top =  cord_thres[l][1] * shape[0]
                    right = cord_thres[l][2] * shape[1]
                    bottom = cord_thres[l][3] * shape[0]
                    width = right - left 
                    height = bottom - top 
                    final_detections.append((int(label), cord_thres[l][4], (left,top,width,height)))
    final_detections = non_max_suppression_fast(final_detections, .7)
    result = draw_boxes(final_detections, img)
    result = draw_boxes([], img, draw_dims=True, dims=d.precise_dims, dims_color=(255,0,255))
    cv2.imwrite(f'dark_walmart_result_{d.method}.jpg', result)

def precise_test(file_name):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    img_paths = ["D:\\Downloads\\camera_8_trim_Moment.jpg", "D:\\Downloads\\DI3P2L52GBCURMVJEWQTKXOAOY.jpg", "D:\\VSProjects]\\DepthRewriteTorch\\scrapbook_test.jpg" , "D:\\VSProjects]\\DepthRewriteTorch\\1000x1000_test.jpg", "D:\\VSProjects]\\DepthRewriteTorch\\1200x1200.jpg"]
    img = cv2.imread("D:\\VSProjects]\\DepthRewriteTorch\\slice_test_1.jpeg")
    shape = img.shape

    d = DepthSlicer('precise', img, .90, .2, 1000, True, square_size=50)
    dims = d.dims

    results = model(img)
    results.save()

    final_detections = []
    for index, i in enumerate(results.xyxyn):
        labels, cord_thres = i[:, -1].numpy(), i[:, :-1].numpy()
        if (index < len(dims)):
            for l, label in enumerate(labels): 
                if label == 0: 
                    left = dims[index][0] + cord_thres[l][0] * (dims[index][1] - dims[index][0])
                    top = dims[index][2] + cord_thres[l][1] * (dims[index][3] - dims[index][2])
                    right = dims[index][0] + cord_thres[l][2] * (dims[index][1] - dims[index][0])
                    bottom = dims[index][2] + cord_thres[l][3] * (dims[index][3] - dims[index][2])
                    width = right - left 
                    height = bottom - top 
                    final_detections.append((int(label), cord_thres[l][4], (left,top,width,height)))
        else: 
            for l, label in enumerate(labels): 
                if label == 0: 
                    left = cord_thres[l][0] * shape[1]
                    top =  cord_thres[l][1] * shape[0]
                    right = cord_thres[l][2] * shape[1]
                    bottom = cord_thres[l][3] * shape[0]
                    width = right - left 
                    height = bottom - top 
                    final_detections.append((int(label), cord_thres[l][4], (left,top,width,height)))

    final_detections = non_max_suppression_fast(final_detections, .7)
    result = draw_boxes(final_detections, img, draw_dims=True, dims=d.dims)
    result = draw_boxes([], img, draw_dims=True, dims=d.precise_dims, dims_color=(255,0,255))
    cv2.imwrite(f'dark_walmart_result_{d.method}.jpg', result)