
from depth_slicer import DepthSlicer
from detection_utils import non_max_suppression_fast, draw_boxes
from slice_grid_manager import SliceGridManager

import argparse
import torch 
import cv2
import numpy as np
import signal
import time
import threading

# Globals 
# Video Writer
video_writer = None

def parse_args(): 
    parser = argparse.ArgumentParser(
        'Change params / method with args'
    )
    parser.add_argument('--path', type=str,
                    help='path to video, or directory of videos [TBI]')
    parser.add_argument('--prop_thresh', type=float, help='proportion threshold, value between 0.0 and 1.0', default=0.9)
    parser.add_argument('--depth_thresh', type=float, help='depth threshold, between 0.0 and 1.0. 1.0 is closest.', default=0.2)
    parser.add_argument('--method', type=str, help='Method of image processing', choices=['precise_grid', 'precise'], default='precise')
    parser.add_argument('--slice_side_length', type=int, help='length of slice sides', default=800)
    parser.add_argument('--square_size', type=int, help='side length of squares image is split up into in calibration.', default=50)
    return parser.parse_args()

def write_frame(frame): 
    global video_writer 
    video_writer.write(frame)
    return

def precise_grid_video(file_name, prop_thresh=0.9, depth_thresh=.2, square_size=50):
    global video_writer

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cpu')

    video = cv2.VideoCapture(file_name)
    FPS = int(video.get(cv2.CAP_PROP_FPS))  
    shape = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    outfile = file_name.split('.')[0] + '_result_test.avi'
    video_writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), FPS, shape)  

    ret, init_frame = video.read()

    if ret: 
        d = DepthSlicer('precise_grid', cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB), prop_thresh, depth_thresh, 1000, True, square_size=square_size)
        slice_grid_manager = d.dims
    else: 
        print('Bad Video')
        exit()

    frame_count = 1
    detection_check_count = 1

    while True: 
        start_time = time.time()
        ret, frame = video.read()
        images = []
        if ret: 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            detection_check_count += 1
            if detection_check_count > slice_grid_manager.refresh_interval:
                slices = slice_grid_manager.get_slices(frame, all=True)
                for i, new_slice in enumerate(slices): 
                    new_slice = cv2.resize(new_slice, (640, 640), interpolation=cv2.INTER_LINEAR)
                    images.append(new_slice)
            else: 
                slices = slice_grid_manager.get_slices(frame, all=False)
                for i, new_slice in enumerate(slices): 
                    cv2.imwrite(f'slice_{i}.jpeg', new_slice)
                    new_slice = cv2.resize(new_slice, (640, 640), interpolation=cv2.INTER_LINEAR)
                    images.append(new_slice)
        else: 
            print('Video finished')
            break

        images.append(frame)

        results = model(images)

        final_detections = []
        print(len(results.xyxyn))
        for index, i in enumerate(results.xyxyn):
            # remove .cpu() if you're on gpu
            labels, cord_thres = np.array(i.cpu())[:, -1], np.array(i.cpu())[:, :-1]
            if index < len(results.xyxyn) - 1:
                for l, label in enumerate(labels): 
                    if label == 0: 
                        grid_index = slice_grid_manager.active_grid_indices[index]
                        left, right, top, bottom = slice_grid_manager.grids[grid_index].normal_to_pixel_coords(cord_thres[l])
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
        
        if detection_check_count > slice_grid_manager.refresh_interval:
            detection_check_count = 0
            slice_grid_manager.check_detection()
        
        prev = time.time()
        final_detections = non_max_suppression_fast(final_detections, .5)

        print(f'NMS time: {time.time() - prev}')

        print(f'FPS: {1 / (time.time() - start_time)}')
        print(f'Frame Number: {frame_count}')
        image = draw_boxes(final_detections, frame, draw_dims=True, dims=d.precise_dims, dims_color=(255,0,255))
        video_write_buffer = threading.Thread(target=write_frame, args=(image,))
        video_write_buffer.start()

    video_writer.release()
        

def precise_video(file_name, prop_thresh=0.9, depth_thresh=.2, square_size=50, slice_side_length=800):
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
        d = DepthSlicer('precise', cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB), prop_thresh, depth_thresh, slice_side_length=slice_side_length, regen=True, square_size=square_size)
        dims = d.dims
        print(dims)
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
            new_slice = frame[dim[2]:dim[3],dim[0]:dim[1]]
            new_slice = cv2.resize(new_slice, (640, 640), interpolation=cv2.INTER_LINEAR)
            images.append(new_slice)
        images.append(frame)

        results = model(images)

        final_detections = []
        for index, i in enumerate(results.xyxyn):
            labels, cord_thres = i.cpu()[:, -1].numpy(), i.cpu()[:, :-1].numpy()
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
        image = draw_boxes(final_detections, frame, draw_dims=True, dims=d.dims, dims_color=(255,0,255))
        video_write_buffer = threading.Thread(target=write_frame, args=(image,))
        video_write_buffer.start()

    video_writer.release()

def exit_handler(sigint, frame): 
    global video_writer 
    video_writer.release()
    exit(0)

if __name__ == "__main__":
    args = parse_args()
    signal.signal(signal.SIGINT, exit_handler)
    if args.method == 'precise_grid': 
        precise_grid_video(args.path, args.prop_thresh, args.depth_thresh, args.square_size)
    else: 
        precise_video(args.path, args.prop_thresh, args.depth_thresh, args.square_size, args.slice_side_length)