
from depth_slicer import DepthSlicer
from detection_utils import non_max_suppression_fast, draw_boxes

import argparse
import contextlib
import cProfile, pstats
import io
import torch, torchvision
import torch.cuda.profiler as profiler
import pyprof
import cv2
import numpy as np
import signal
import queue
import time
from threading import Thread
from pstats import SortKey

#################
#    TODO 
#################

# 1. Do os path for all inputs / outputs for OS compatability 
# 2. Try out batch detection 
# 3. Optimize ?? 
# 4. Edge case on precise grid
# 5. Done ? 


# Globals 
# Video Writer
video_writer = None

@contextlib.contextmanager 
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg) 
    try: 
        yield depth 
    finally: 
        torch.cuda.nvtx.range_pop()

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
    parser.add_argument('--grid_width', type=int, help='how many columns in precise grid slices', default=3)
    parser.add_argument('--grid_height', type=int, help='number of rows in precise grid slices', default=4)
    parser.add_argument('--refresh_rate', type=int, help='number of frames between precise grid refreshes', default=50)
    return parser.parse_args()

def write_frame(frame): 
    global video_writer 
    video_writer.write(frame)
    return

def precise_grid_video(args, file_name, prop_thresh=0.9, depth_thresh=.2, square_size=50):
    global video_writer

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else model.to('cpu')
    preprocess = torchvision.transforms.ToTensor()

    video = cv2.VideoCapture(file_name)
    FPS = int(video.get(cv2.CAP_PROP_FPS))  
    shape = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    outfile = file_name.split('.')[0] + '_result_test.avi'
    video_writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), FPS, shape)  

    ret, init_frame = video.read()

    if ret: 
        d = DepthSlicer('precise_grid', cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB), prop_thresh, depth_thresh, args.slice_side_length, True, 
                        square_size=square_size, grid_w=args.grid_width, grid_h=args.grid_height, refresh_rate=args.refresh_rate)
        slice_grid_manager = d.dims
    else: 
        print('Bad Video')
        exit()

    frame_count = 1
    detection_check_count = 1
    start_time = time.time()

    while True: 
        ret, frame = video.read()
        frame_time = time.time()
        images = []
        if ret: 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            detection_check_count += 1
            if detection_check_count > slice_grid_manager.refresh_interval:
                slices = slice_grid_manager.get_slices(frame, all=True)
                for i, new_slice in enumerate(slices): 
                    new_slice = cv2.resize(new_slice,  (640, 640), interpolation=cv2.INTER_LINEAR)
                    images.append(new_slice)
            else: 
                slices = slice_grid_manager.get_slices(frame, all=False)
                for i, new_slice in enumerate(slices): 
                    cv2.imwrite(f'slice_{i}.jpeg', new_slice)
                    new_slice =cv2.resize(new_slice,  (640, 640), interpolation=cv2.INTER_LINEAR)
                    images.append(new_slice)
        else: 
            print('Video finished')
            break

        images.append(cv2.resize(frame,  (640, 640), interpolation=cv2.INTER_LINEAR))

        with torch.no_grad():
            results = model(images)

        final_detections = []
        for index, i in enumerate(results.xyxyn):
            labels, cord_thres = np.array(i)[:, -1], np.array(i.cpu())[:, :-1]
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

        final_detections = [detection for detection in final_detections if detection[1] > .1]
        final_detections = non_max_suppression_fast(final_detections, .5)

        print(f'Frame Number: {frame_count}')
        image = draw_boxes(final_detections, frame, draw_dims=True, dims=d.precise_dims, dims_color=(255,0,255))
        video_write_buffer = Thread(target=write_frame, args=(image,))
        video_write_buffer.start()
        print(f'FPS: {1/(time.time()-frame_time)}')
    print(f'Average FPS: {frame_count / (time.time()-start_time):.2f}')
    video_writer.release()

global_cap = None
frame_queue = None 
image_queue = None
frame_count = 1

def video_capture(frame_queue, image_queue): 
    global global_cap

    while global_cap.isOpened(): 
        ret, frame = global_cap.read()
        if not ret: 
            image_queue.put(None)
            frame_queue.put(None)
            break 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_queue.put(frame)
        image_queue.put(frame_rgb)

def precise_pre_process(frame, dims):
    images = []
    for i, dim in enumerate(dims):
        new_slice = frame[dim[2]:dim[3],dim[0]:dim[1]]
        new_slice = cv2.resize(new_slice, (640, 640), interpolation=cv2.INTER_LINEAR)
        images.append(new_slice)
    images.append(cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR))
    return images

def inference(model, image_queue, detections_queue, fps_queue, dims): 
    global global_cap
    while global_cap.isOpened(): 
        frame = image_queue.get() 
        if frame is None: 
            break 
        prev_time = time.time() 

        images = precise_pre_process(frame, dims)
        result = model(images)

        detections_queue.put(result)
        fps = float(1/(time.time()-prev_time))
        fps_queue.put(fps)
        print(f'FPS: {fps}')
    
def drawing(frame_queue, detections_queue, dims, hw_dims, fps_queue):
    global video_writer, global_cap, frame_count

    while global_cap.isOpened(): 
        frame = frame_queue.get()
        if frame is None: break
        results = detections_queue.get()
        fps = fps_queue.get()
        bboxes, scores, labels = precise_post_process(results, hw_dims)
        image = draw_boxes(bboxes, scores, labels, frame, draw_dims=True, dims=dims, dims_color=(255,0,255))
        frame_count += 1
        write_frame(image)
        if cv2.waitKey(int(fps)) == 27:
            break
    global_cap.release()
    video_writer.release()

def precise_post_process(results, hw_dims): 
    cords = []
    scores = []
    labels = []
    for i, result in enumerate(results.xyxy):
        # get labels, cords, scores
        labels.append(result[:, -1])
        cord_thres = result[:, :4]
        score = result[:, 4:-1]

        # multiply coordinates to original pixel space (l, t, r, b)
        cord_thres[:, :2] = cord_thres[:, :2] + hw_dims[i, 1]
        cord_thres[:, 2:-1] = cord_thres[:, 2:-1] + hw_dims[i, 0]

        cords.append(cord_thres)
        scores.append(score)
    
    all_cords = torch.cat(cords)
    all_labels = torch.cat(labels)
    all_scores = torch.cat(scores).view(-1)

    nms_mask = torchvision.ops.boxes.batched_nms(all_cords, all_scores, all_labels, iou_threshold=0.7)
    
    bboxes = all_cords[nms_mask].cpu().numpy()
    scores = all_scores[nms_mask].cpu().numpy()
    labels = all_labels[nms_mask].cpu().numpy()
    return bboxes, scores, labels

def precise_video(args, file_name, prop_thresh=0.9, depth_thresh=.2, square_size=50, slice_side_length=800):
    global video_writer, global_cap, frame_queue, image_queue, frame_count

    pr = cProfile.Profile()
    pr.enable()

    model_dtype = torch.float16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(torch.version.cuda)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval().to(device)
    model.conf = .2 
    model.iou = .25

    global_cap = cv2.VideoCapture(file_name)
    FPS = int(global_cap.get(cv2.CAP_PROP_FPS))  
    shape = (int(global_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(global_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    outfile = file_name.split('.')[0] + '_result_precise.avi'
    video_writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'XVID'), FPS, shape)  

    ret, init_frame = global_cap.read()

    if ret: 
        d = DepthSlicer('precise', cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB), prop_thresh, depth_thresh, slice_side_length=slice_side_length, regen=True, square_size=square_size)
        dims = d.dims
        hw_dims = [] 
        for dim in dims: 
            hw_dims.append([dim[2], dim[0]])
        hw_dims.append([0, 0])
    else: 
        print('Bad Video')
        exit()

    # turn dims to tensor
    hw_dims = torch.tensor(hw_dims, dtype=model_dtype, device=device)

    frame_count = 1
    start_time = time.time()
    
    frame_queue = queue.Queue()
    image_queue = queue.Queue(maxsize=1)
    detections_queue = queue.Queue(maxsize = 1)
    fps_queue = queue.Queue(maxsize=1)

    capture_thread = Thread(target=video_capture, args=(frame_queue,image_queue))
    inference_thread = Thread(target=inference, args=(model, image_queue, detections_queue, fps_queue, dims))
    drawing_thread = Thread(target=drawing, args=(frame_queue, detections_queue, dims, hw_dims, fps_queue))

    capture_thread.start()
    inference_thread.start()
    drawing_thread.start()

    capture_thread.join()
    inference_thread.join()
    drawing_thread.join()

    print(f'AVG FPS: {frame_count/(time.time()-start_time):2f}')

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    ps.dump_stats('./stats/cprofile')
    out_stream = open(f'./stats/stats_{args.method}', 'w')
    ps = pstats.Stats('./stats/cprofile', stream=out_stream)
    ps.strip_dirs().sort_stats('cumulative').print_stats()
    video_writer.release()

def exit_handler(sigint, frame):
    global global_cap, video_writer, frame_queue, image_queue
    print("exiting")
    if frame_queue:
        frame_queue.put(None)
    if image_queue:
        image_queue.put(None)
    if global_cap:
        global_cap.release()
    if video_writer:
        video_writer.release()
    time.sleep(1)
    print("exiting sucessfully")

if __name__ == "__main__":
    args = parse_args()
    signal.signal(signal.SIGINT, exit_handler)
   
    if args.method == 'precise_grid': 
        precise_grid_video(args, args.path, args.prop_thresh, args.depth_thresh, args.square_size)
    else: 
        precise_video(args, args.path, args.prop_thresh, args.depth_thresh, args.square_size, args.slice_side_length)