
from depth_slicer import DepthSlicer
from detection_utils import non_max_suppression_fast, draw_boxes

import torch 
import cv2
import numpy as np
import time
import threading

# Globals 
# Video Writer
video_writer = None

def write_frame(frame): 
    global video_writer 
    video_writer.write(frame)
    return

def precise_grid_video(file_name, prop_thresh=0.9, depth_thresh=.2, square_size=50):
    global video_writer

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    video = cv2.VideoCapture(file_name)
    FPS = int(video.get(cv2.CAP_PROP_FPS))  
    shape = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    outfile = file_name.split('.')[0] + '_result_precise_grid.avi'
    video_writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), FPS, shape)  

    ret, init_frame = video.read()

    if ret: 
        d = DepthSlicer('precise_grid', cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB), prop_thresh, depth_thresh, 1000, True, square_size=square_size)
        dims = d.dims
    else: 
        print('Bad Video')
        exit()

    frame_count = 1

    while True: 
        start_time = time.time()
        ret, frame = video.read()
        if ret: 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
        else: 
            print('Video finished')
            break

        images = []
        for i, dim in enumerate(dims): 
            new_slice = dim.make_grid_image(frame)
            new_slice = cv2.resize(new_slice, (640, 640), interpolation=cv2.INTER_LINEAR)
            images.append(new_slice)
        images.append(frame)

        results = model(images)

        final_detections = []
        for index, i in enumerate(results.xyxyn):
            # comment out .cpu() if you're on cpu
            labels, cord_thres = np.array(i.cpu())[:, -1], np.array(i.cpu())[:, :-1]
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
                        left = cord_thres[l][0] * shape[0]
                        top =  cord_thres[l][1] * shape[1]
                        right = cord_thres[l][2] * shape[0]
                        bottom = cord_thres[l][3] * shape[1]
                        width = right - left 
                        height = bottom - top 
                        final_detections.append((int(label), cord_thres[l][4], (left,top,width,height)))
        final_detections = non_max_suppression_fast(final_detections, .7)

        print(f'FPS: {1 / (time.time() - start_time)}')
        print(f'Frame Number: {frame_count}')
        image = draw_boxes(final_detections, frame, draw_dims=True, dims=d.precise_dims, dims_color=(255,0,255))
        video_write_buffer = threading.Thread(target=write_frame, args=(image,))
        video_write_buffer.start()

    video_writer.release()
        

def precise_video(file_name, prop_thresh=0.9, depth_thresh=.2, square_size=50, lower_bound=1000):
    global video_writer

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    video = cv2.VideoCapture(file_name)
    FPS = int(video.get(cv2.CAP_PROP_FPS))  
    shape = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    outfile = file_name.split('.')[0] + '_result_precise.avi'
    video_writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), FPS, shape)  

    ret, init_frame = video.read()

    if ret: 
        d = DepthSlicer('precise', cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB), prop_thresh, depth_thresh, slice_side_length=lower_bound, regen=True, square_size=square_size)
        dims = d.dims
    else: 
        print('Bad Video')
        exit()

    frame_count = 1

    while True: 
        start_time = time.time()
        ret, frame = video.read()
        if ret: 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
        else: 
            print('Video finished')
            break

        images = []
        for i, dim in enumerate(dims): 
            prev = time.time()
            new_slice = dim.make_grid_image(frame)
            cv2.imwrite(f'slice_test_{i}.jpeg', new_slice)
            new_slice = cv2.resize(new_slice, (640, 640), interpolation=cv2.INTER_LINEAR)
            images.append(new_slice)
        images.append(frame)

        results = model(images)

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
                        left = cord_thres[l][0] * shape[0]
                        top =  cord_thres[l][1] * shape[1]
                        right = cord_thres[l][2] * shape[0]
                        bottom = cord_thres[l][3] * shape[1]
                        width = right - left 
                        height = bottom - top 
                        final_detections.append((int(label), cord_thres[l][4], (left,top,width,height)))
        final_detections = non_max_suppression_fast(final_detections, .7)

        print(f'FPS: {1 / (time.time() - start_time)}')
        print(f'Frame Number: {frame_count}')
        image = draw_boxes(final_detections, frame, draw_dims=True, dims=d.precise_dims, dims_color=(255,0,255))
        video_write_buffer = threading.Thread(target=write_frame, args=(image,))
        video_write_buffer.start()

    video_writer.release()

if __name__ == "__main__":
    precise_grid_video('/content/test_camera_8_30fps.mp4')