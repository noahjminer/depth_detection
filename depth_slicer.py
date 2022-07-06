from concurrent.futures import process
from monodepth2.test_simple import create_depth_image_from_frame

from slice_grid_manager import SliceGridManager
from slice_grid import SliceGrid
import torchvision
import torch
import cv2 
import numpy as np
import math
import os 
import time


class DepthSlicer: 
    """
        A class for depth image generation and slice calculation

        ... 

        Attributes: 
        -----------
        method : str 
            specifies slicing method <simple | precise> 
        init_frame : cv2 image object
            Initial frame of stream / video, or just a single image
        proportion_thresh : float
            between 0.0 and 1.0, specifies lower bound on pixels past distance threshold
            over total pixels in checked slice
        dist_thresh : float 
            between 0.0 and 1.0, specifies lower bound on distance. 0.0 is farther, and 1.0 is 
            closer  
        """
    def __init__(self, method, init_frame, proportion_thresh, dist_thresh, slice_side_length=608, regen_depth=True, regen_dims=False, square_size=100,
                grid_w=3, grid_h=4, refresh_rate=50, device=None): 
        """
        Sets attributes and generates depth image if regen=True

        ... 

        Parameters: 
        -----------
        method : str 
            specifies slicing method <simple | precise> 
        init_frame : cv2 image object
            Initial frame of stream / video, or just a single image
        proportion_thresh : float
            between 0.0 and 1.0, specifies lower bound on pixels past distance threshold
            over total pixels in checked slice
        dist_thresh : float 
            between 0.0 and 1.0, specifies lower bound on distance. 0.0 is farther, and 1.0 is 
            closer 
        regen : bool 
            if True, regenerates depth image. If false, looks for existing depth image
        dims : list of lists [xmin, xmax, ymin, ymax]
            list of dimensions for slicing 
        slice_side_length : int
            minimum slice side length for each slice (in simple method). default is 608
        square_size : int 
            side length of grid squares image is divided into 
        tile_w : int 
            number of columns in SliceGrids
        tile_h : int 
            number of rows in SliceGrids
        refresh_rate : int 
            number of frames between refreshes in SliceGridManager
        """
        self.dist_thresh = dist_thresh
        self.square_size = square_size
        self.proportion_thresh = proportion_thresh 
        self.method = method
        self.frame = init_frame
        self.dims = []
        self.precise_dims = []
        self.slice_side_length = slice_side_length
        self.path = os.path.join(os.getcwd(), "frame_disp.jpeg")
        self.grid_w = grid_w 
        self.grid_h = grid_h 
        self.refresh_rate = refresh_rate
        self.tlhw_dims = []
        self.device = device
        if regen_depth:
            self.generate_depth_image(init_frame)

        if regen_dims:
            self.dims = self.calculate_dims()
        else:
            self.dims = self.read_slice_file()
            print(self.dims)
            if not self.dims: 
                self.dims = self.calculate_dims()
            else:
                tlhw_dims = []
                for dim in self.dims: 
                    tlhw_dims.append([dim[2], dim[0], dim[3]-dim[2], dim[1]-dim[0]])
                tlhw_dims.append([0, 0, self.frame.shape[0], self.frame.shape[1]])
                self.tlhw_dims = torch.tensor(tlhw_dims, dtype=torch.float16, device=self.device)
                
        
    def generate_depth_image(self, frame) -> None:
        """
        Generates depth image from supplied frame

        ...

        Parameters: 
        -----------
        frame : cv2 image object
            frame to be used in depth image generation
        """
        create_depth_image_from_frame(frame)
    

    def calculate_dims(self) -> list: 
        """
        Hub for different methods for calculating slice dimensions

        ... 
        
        Parameters:
        None
        """
        if not os.path.exists('frame_disp.jpeg'): 
            self.generate_depth_image(self.frame)

        prev_time = time.time()
        if self.method == 'simple':
            dims = self.calculate_simple_dims()
        elif self.method == 'precise': 
            dims = self.calculate_precise_dims()
        elif self.method == 'precise_grid':
            dims = self.calculate_precise_grid_dims()
        else: 
            dims = self.calculate_mask_dims()
        print('---------')
        print(f'Dimension calculation done in {time.time() - prev_time}')
        print('---------')
        self.write_slice_file(dims)
        return dims

    
    def create_depth_mask(self, image, dim, masked=False) -> bool:
        """
        Loops through pixels within dimension in image, checking color values with thresholds

        Parameters
        ... 
        image : opencv2 image object
            Depth map generated on init
        dim : list of ints
            Dimension (xmin, xmax, ymin, ymax)
        """
        count = 0
        for i in range(dim[2], dim[3]):
            for j in range(dim[0], dim[1]):
                dist = image[i][j][0] * 0.0039
                if not masked and dist < self.dist_thresh:
                    count += 1
                elif masked and dist > 0:
                    count += 1
        count = count / ((dim[3] - dim[2]) * (dim[1] - dim[0]))
        if count > self.proportion_thresh:
            return True
        return False


    def calculate_mask_dims(self):
        depth_img = cv2.imread(self.path, 0)
        _, mask = cv2.threshold(depth_img, thresh=50, maxval=30, type=cv2.THRESH_BINARY_INV)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3 channel mask
        masked_frame = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        cv2.imwrite('masked_frame.jpg', masked_frame)

        # find local minimums of slices
        graph, image_shape = self.generate_depth_graph_from_mask(mask3, self.square_size)

        blobs = np.asarray(self.connect_squares_into_islands(graph), dtype=object)
                
        # Process connected blobs into one box
        dims = self.create_bounding_box_for_islands_mask(blobs, image_shape, self.square_size)
        
        processed_dims, _, _ = self.divide_island_boxes(blobs, dims, image_shape, self.slice_side_length)

        tlhw_dims = []
        for dim in processed_dims: 
            tlhw_dims.append([dim[2], dim[0], dim[3]-dim[2], dim[1]-dim[0]])
        tlhw_dims .append([0, 0, self.frame.shape[0], self.frame.shape[1]])

        self.tlhw_dims = torch.tensor(tlhw_dims, dtype=torch.float16, device=self.device)
        # Process dims by splitting up bigger boxes 
        return processed_dims

    def calculate_precise_dims(self):
        """
        Calculates more precise dimensions by creating a graph of elligible portions of the image

        ...

        Parameters:
        None
        """
        graph, image_shape = self.generate_depth_graph(0, self.square_size)
        # Connect islands into one list
        blobs = self.connect_squares_into_islands(graph)
        sorted_blobs = [sorted(blob, key=lambda coord: coord[0]) for blob in blobs]
        blobs_2d = []
        for blob in sorted_blobs:
            blobs_rows = []
            blob_row = []
            curr_row = blob[0][0]
            for coord in blob:
                if coord[0] == curr_row: 
                    blob_row.append(coord)
                else:
                    blobs_rows.append(blob_row)
                    curr_row = coord[0]
                    blob_row = [coord]
            blobs_2d.append(blobs_rows)
        
        for i, blob in enumerate(blobs_2d): 
            blobs_2d[i] = [sorted(row, key=lambda coord: coord[1]) for row in blob]
                
        # Process connected blobs into one box
        dims = self.create_bounding_box_for_islands(blobs_2d, image_shape, self.square_size)

        processed_dims, _, _ = self.divide_island_boxes(blobs, dims, image_shape, self.slice_side_length)

        tlhw_dims = []
        for dim in processed_dims: 
            tlhw_dims.append([dim[2], dim[0], dim[3]-dim[2], dim[1]-dim[0]])
        tlhw_dims.append([0, 0, self.frame.shape[0], self.frame.shape[1]])

        self.tlhw_dims = torch.tensor(tlhw_dims, dtype=torch.float16, device=self.device)
        
        # Process dims by splitting up bigger boxes 
        return processed_dims

    def calculate_precise_grid_dims(self): 
        """
        Calculates precise grid dimensions. Grid meaning a collage of slices taken from the image and put into one for processing.
        """
        # calculate precise dimensions
        graph, image_shape = self.generate_depth_graph(0, self.square_size)
        
        # Connect islands into one list
        blobs = self.connect_squares_into_islands(graph)
        
        # Process connected blobs into one box
        dims = self.create_bounding_box_for_islands(blobs, image_shape, self.square_size)
        
        # Process dims by splitting up bigger boxes 
        processed_dims, max_w, max_h = self.divide_island_boxes(dims, image_shape, self.slice_side_length)

        self.processed_dims = processed_dims
        
        tile_width = int(max_w) 
        tile_height = int(max_h)
        
        slice = SliceGrid(tile_width, tile_height, w=self.grid_w, h=self.grid_h)
        slices = [slice]
        for dim in processed_dims: 
            if slice.full: 
                slice.create_empty_image()
                slice = SliceGrid(tile_width, tile_height, w=self.grid_w, h=self.grid_h)
                slices.append(slice)
            if dim[3] - dim[2] > tile_height: dim[3] -= dim[3] - dim[2] - tile_height
            if dim[1] - dim[0] > tile_width: dim[1] -= dim[1] - dim[0] - tile_width
            slice.insert_tile(dim)
        slice.create_empty_image()

        slice_manager = SliceGridManager(slices, refresh_interval=self.refresh_rate)
        
        return slice_manager

    def precise_post_process(self, results):
        cords = []
        scores = []
        labels = []
        for i, result in enumerate(results.xyxyn):
            # get labels, cords, scores
            labels.append(result[:, -1])
            cord_thres = result[:, :4]
            score = result[:, 4:-1]

            # multiply coordinates to original pixel space (l, t, r, b)
            cord_thres[:, 0] = cord_thres[:, 0] * self.tlhw_dims[i, 3] + self.tlhw_dims[i, 1]
            cord_thres[:, 2] = cord_thres[:, 2] * self.tlhw_dims[i, 3] + self.tlhw_dims[i, 1]
            cord_thres[:, 1] = cord_thres[:, 1] * self.tlhw_dims[i, 2] + self.tlhw_dims[i, 0]
            cord_thres[:, 3] = cord_thres[:, 3] * self.tlhw_dims[i, 2] + self.tlhw_dims[i, 0]

            cords.append(cord_thres)
            scores.append(score)
        
        all_cords = torch.cat(cords)
        all_labels = torch.cat(labels)
        all_scores = torch.cat(scores).view(-1)

        #score_mask = all_scores > .2

        nms_mask = torchvision.ops.boxes.batched_nms(all_cords, all_scores, all_labels, iou_threshold=.5)
        # nms_mask = torchvision.ops.boxes.batched_nms(all_cords[score_mask], all_scores[score_mask], all_labels[score_mask], iou_threshold=.5)
        
        bboxes = all_cords[nms_mask]
        scores = all_scores[nms_mask]
        labels = all_labels[nms_mask]

        # Find overlap from big image and overwrite boxes underneath
        now = time.time()
        overlap_mask = self.intersect_suppression(bboxes)
        print(f'took {time.time() - now}')

        return bboxes[overlap_mask].cpu().numpy(), scores[overlap_mask].cpu().numpy(), labels[overlap_mask].cpu().numpy()

    def intersect_suppression(self, coords):     
        x1 = coords[:, 0]
        y1 = coords[:, 1]
        x2 = coords[:, 2]
        y2 = coords[:, 3]
        areas = torch.mul(x2 - x1, y2 - y1)
        area_idxs = torch.argsort(areas, dim=0, descending=False)
        iarea_mask = area_idxs > 0

        for i, idx in enumerate(area_idxs):
            xx1 = torch.maximum(x1[idx], x1[area_idxs[i+1:]])
            yy1 = torch.maximum(y1[idx], y1[area_idxs[i+1:]])
            xx2 = torch.minimum(x2[idx], x2[area_idxs[i+1:]])
            yy2 = torch.minimum(y2[idx], y2[area_idxs[i+1:]])
            iarea = torch.mul(xx2 - xx1, yy2 - yy1)
            iarea_mask[i] = not torch.any(torch.abs(iarea - areas[idx]) < 300)
        
        return area_idxs[iarea_mask]

    def divide_island_boxes(self, blobs, dims, image_shape, lower_bound=800) -> list:
        """
        Divides larger bounding boxes into smaller divisions, depending on lower_bound. 

        ... 

        Parameters: 
        -----------
        dims : list of int dimensions [xmin, xmax, ymin, ymax, blob_index]
            dimensions of larger bounding boxes
        blobs : list of lists with grid squares
            Used for some cleanup in dim division
        image_shape : (height, width)
            Shape of image, used for bounds checking 
        lower_bound : int 
            value with which bounding boxes will be divided
        """
        processed_dims = [] 
        max_w = 0
        max_h = 0
        xlower = lower_bound 
        ylower = lower_bound + (.2 * lower_bound)
        for index, dim in enumerate(dims): 
            w = dim[1] - dim[0]
            h = dim[3] - dim[2]

            # if both are less then just add it in 
            if w < xlower and h < lower_bound:
                processed_dims.append(dim)
                continue
    
            # want each stack to be ~800-1200 px 
            horiz_segments = int(math.floor(w / xlower))
            vert_segments = int(math.floor(h / lower_bound)) 
            if not horiz_segments: horiz_segments += 1
            if not vert_segments: vert_segments += 1
            horiz_remainder = w % xlower
            vert_remainder = h % ylower

            segment_width = xlower + horiz_remainder / horiz_segments if horiz_segments > 1 else w
            segment_height = ylower + vert_remainder / vert_segments if vert_segments > 1 else h

            woverlap = 0.1 * segment_width # NEEDS PARAMATERIZATION
            hoverlap = 0.1 * segment_height

            max_w = segment_width + 2 * woverlap if segment_width + 2 * woverlap > max_w else max_w
            max_h = segment_height + 2 * hoverlap if segment_height + 2 * hoverlap > max_h else max_h
            
            for i in range(1, horiz_segments+1): 
                for j in range(1, vert_segments+1): 
                    new_dim = [0,0,0,0]
                    new_dim[0] = dim[0] + (i-1) * segment_width
                    new_dim[1] = dim[0] + (i-1) * segment_width + segment_width
                    new_dim[2] = dim[2] + (j-1) * segment_height
                    new_dim[3] = dim[2] + (j-1) * segment_height + segment_height

                    local_mask_x = np.ma.getmask(np.ma.masked_inside(blobs[index][:, 1], new_dim[0] // self.square_size, new_dim[1] // self.square_size))
                    local_mask_y = np.ma.getmask(np.ma.masked_inside(blobs[index][:, 0], new_dim[2] // self.square_size, new_dim[3] // self.square_size))
                    local_mask = np.logical_and(local_mask_x, local_mask_y)

                    masked_blob = blobs[index][local_mask]
                    mins = np.amin(masked_blob, axis=0) * self.square_size
                    maxs = np.amax(masked_blob, axis=0) * self.square_size

                    if new_dim[0] < mins[1]: new_dim[0] = mins[1] 
                    if new_dim[1] > maxs[1]: new_dim[1] = maxs[1] 
                    if new_dim[2] < mins[0]: new_dim[2] = mins[0] 
                    if new_dim[3] > maxs[0]: new_dim[3] = maxs[0]

                    dim_w = new_dim[1] - new_dim[0]
                    dim_h = new_dim[3] - new_dim[2]

                    # check if both are good, then add half of overlap 
                    if horiz_segments > 1 or dim_w < self.slice_side_length:
                        # Change overlap to match side length instead
                        if dim_w < self.slice_side_length: woverlap = .5 * (self.slice_side_length - dim_w)

                        if new_dim[0] - woverlap > 0 and new_dim[1] + woverlap < image_shape[1] - 1:
                            new_dim[0] = new_dim[0] - woverlap
                            new_dim[1] = new_dim[1] + woverlap
                        elif new_dim[1] + 2 * woverlap < image_shape[1] - 1: 
                            new_dim[0] = 0
                            new_dim[1] = new_dim[1] + 2 * woverlap
                        elif new_dim[0] - 2 * woverlap > 0: 
                            new_dim[1] = image_shape[1] - 0
                            new_dim[0] = new_dim[0] - 2 * woverlap
                    

                    if vert_segments > 1 or dim_h < self.slice_side_length:
                        if dim_h < self.slice_side_length: hoverlap = .5 * (self.slice_side_length - dim_h)

                        if new_dim[2] - hoverlap > 0 and new_dim[3] + hoverlap < image_shape[0] - 1:
                            new_dim[2] = new_dim[2] - hoverlap
                            new_dim[3] = new_dim[3] + hoverlap
                        elif new_dim[3] + 2 * hoverlap < image_shape[0] - 1: 
                            new_dim[2] = 0
                            new_dim[3] = new_dim[3] + 2 * hoverlap
                        elif new_dim[2] - 2 * hoverlap > 0: 
                            new_dim[3] = image_shape[0] - 0
                            new_dim[2] = new_dim[2] - 2 * hoverlap

                    new_dim = [int(math.floor(num)) for num in new_dim]

                    # check local min / max 
                    # want to get coords > min and less than max

                    new_dim.append(index)
                    processed_dims.append(new_dim)
        return processed_dims, max_w, max_h


    def generate_depth_graph(self, lower_bound, square_size): 
        """
        Generates a list of grid coordinates for areas of the image that pass the distance threshold. 
        Area size is square_size * square_size.

        Returns list of coordinates : [y,x] and image_shape (height, width) 

        ... 

        Parameters: 
        ----------
        lower_bound : int 
            Used to offset y start in case portion of the image does not need to be considered. Although, usually 0.
        """
        image = cv2.imread(self.path)
        image_shape = image.shape

        graph = []
        for y in range(lower_bound, image_shape[0], square_size):
            ymax = image_shape[0] - 1 if y + square_size > image_shape[0] else y + square_size
            for x in range(0, image_shape[1], square_size): 
                xmax = image_shape[1] - 1 if x + square_size > image_shape[1] else x + square_size 
                # depth mask
                if self.create_depth_mask(image, [x, xmax, y, ymax]): 
                    graph.append((y // square_size, x // square_size))
        return graph, image_shape
    

    def generate_depth_graph_from_mask(self, image, square_size):
        image_shape = image.shape
        graph = []
        for y in range(0, image_shape[0], square_size):
            ymax = image_shape[0] - 1 if y + square_size > image_shape[0] else y + square_size
            for x in range(0, image_shape[1], square_size):
                xmax = image_shape[1] - 1 if x + square_size > image_shape[1] else x + square_size 
                # depth mask
                if self.create_depth_mask(image, [x, xmax, y, ymax], True): 
                    graph.append((y // square_size, x // square_size))
        return graph, image_shape


    def connect_squares_into_islands(self, graph) -> list: 
        """
        Connects neighboring squares of graph into a blob. returns a list of blobs

        Parameters: 
        -----------
        graph : list of coordinates [y, x]
            Graph previously generated in generate_depth_graph
        """
        blobs = []
        for node in graph:
            graph.remove(node)
            blob = [node]
            neighbors = self.get_neighbors(node, graph)
            while len(neighbors) > 0: 
                neighbor = neighbors.pop()
                if neighbor in blob: continue
                blob.append(neighbor)
                if graph.count(neighbor) > 0:
                    graph.remove(neighbor)
                    neighbors = neighbors + self.get_neighbors(neighbor, graph)
            blobs.append(blob)
        return blobs

    
    def create_bounding_box_for_islands(self, blobs, image_shape, square_size) -> list:
        """
        Creates a bounding box that encompasses each blob in blobs by finding the max and min of each blob's x and y

        ... 

        Parameters: 
        -----------
        blobs : list of lists of graph coordinates [y, x]
            Each blob in blobs is an island to be connected
        image_shape : tuple (height, width)
            The shape of the video / image. Used for bounds checking
        square_size : int 
            Length of square side. Multiplied by grid coordinate values to get pixel coordinates
        """
        dims = []
        for blob in blobs: 
            xmin = 100
            ymin = 100
            xmax = 0
            ymax = 0

            for y, row in enumerate(blob):
                row_cpy = row
                for i, coord in enumerate(row_cpy): 
                    if coord[0] < ymin: ymin = coord[0]
                    if coord[0] > ymax: ymax = coord[0]
                    if coord[1] < xmin: xmin = coord[1]
                    if coord[1] > xmax: xmax = coord[1]
                    xl = coord[1] * square_size
                    xr = image_shape[1] - 1 if coord[1] * square_size + square_size > image_shape[1] else coord[1] * square_size + square_size 
                    yt = coord[0] * square_size
                    yb = image_shape[0] - 1 if coord[0] * square_size + square_size > image_shape[0] else coord[0] * square_size + square_size 
                    self.precise_dims.append((xl, xr, yt, yb))
                blob[y] = row_cpy

            xmin *= square_size
            ymin *= square_size
            ymax = image_shape[0] - 1 if ymax * square_size + square_size > image_shape[0] else ymax * square_size + square_size
            xmax = image_shape[1] - 1 if xmax * square_size + square_size > image_shape[1] else xmax * square_size + square_size 
            dims.append((xmin, xmax, ymin, ymax)) 
        return dims


    def create_bounding_box_for_islands_mask(self, blobs, image_shape, square_size) -> list:
        """
        Creates a bounding box that encompasses each blob in blobs by finding the max and min of each blob's x and y

        ... 

        Parameters: 
        -----------
        blobs : list of lists of graph coordinates [y, x]
            Each blob in blobs is an island to be connected
        image_shape : tuple (height, width)
            The shape of the video / image. Used for bounds checking
        square_size : int 
            Length of square side. Multiplied by grid coordinate values to get pixel coordinates
        """
        dims = []
        for blob in blobs: 
            mins = np.amin(blob, axis=0)
            maxs = np.amax(blob, axis=0)

            xmin = mins[1]
            xmax = maxs[1]
            ymin = mins[0]
            ymax = maxs[0]

            xmin *= square_size
            ymin *= square_size
            ymax = max(0, min(ymax * square_size + square_size, image_shape[0] - 1))
            xmax = max(0, min(xmax * square_size + square_size, image_shape[1] - 1))
    
            dims.append((xmin, xmax, ymin, ymax)) 
        return dims

            
    def find_min_max_of_blob(self, blob, square_size, image_shape, dim=None) -> list: 
        """
        Finds the min and max for x and y in a blob of grid tiles

        ...

        Parameters: 
        blob: list[tuple(int, int)]
            Grid coordinates
        square_size: int
            Used to convert grid coordinates to image coordinates
        image_shape: tuple(int, int)
            Size of original image (h, w)
        dim: list[xmin, xmax, ymin, ymax, blob_index]
            If set, only checks grid squares within dimension
            default None
        """
        xmin = 100
        ymin = 100
        xmax = 0
        ymax = 0
        for coord in blob: 
            img_coord_x = coord[0] * square_size
            img_coord_y = coord[1] * square_size
            if dim is not None: 
                if img_coord_x < dim[1] or img_coord_x > dim[1] or img_coord_y > dim[2] or img_coord_y < dim[3]:
                    continue

            # FOR VISUALIZING GRID 
            xl = coord[1] * square_size
            xr = image_shape[1] - 1 if coord[1] * square_size + square_size > image_shape[1] else coord[1] * square_size + square_size 
            yt = coord[0] * square_size
            yb = image_shape[0] - 1 if coord[0] * square_size + square_size > image_shape[0] else coord[0] * square_size + square_size 
            self.precise_dims.append((xl, xr, yt, yb))
            # END 
            if coord[0] < ymin: ymin = coord[0]
            if coord[1] < xmin: xmin = coord[1]
            if coord[0] > ymax: ymax = coord[0]
            if coord[1] > xmax: xmax = coord[1]
        xmin *= square_size
        ymin *= square_size
        ymax = image_shape[0] - 1 if ymax * square_size + square_size > image_shape[0] else ymax * square_size + square_size
        xmax = image_shape[1] - 1 if xmax * square_size + square_size > image_shape[1] else xmax * square_size + square_size 
        return xmin, xmax, ymin, ymax 


    def get_neighbors(self, coord, graph) -> list: 
        """
        Returns neighboring grid tiles present in graph

        ... 

        Parameters: 
        coord: tuple(y, x)
            Grid coordinate of tile
        graph: list[tuple(y,x)]
            List of all elligible tiles
        """
        x = coord[1]
        y = coord[0]
        neighbors = []
        for yn in range(-1,2): 
            for xn in range(-1,2):
                if not yn and not xn or x+xn<0 or x+yn<0: continue
                if (y + yn, x + xn) in graph: 
                    neighbors.append((y + yn, x + xn)) 
        return neighbors


    def write_slice_file(self, dims) -> None:
        """
        Writes slice file to base directory. 

        ...

        Parameters: 
        dims : list of lists of ints
            dimensions calculated on init
        """
        with open(os.path.join(os.getcwd(), 'dims.txt'), 'w') as outfile:
            for dim in dims:
                outfile.write(str(dim[0]) + ' ' + str(dim[1]) + ' ' 
                            + str(dim[2]) + ' ' + str(dim[3]) + '\n')
    

    def read_slice_file(self) -> list: 
        """
        Reades slice file from local directory

        ...

        Parameters: 
        None
        """
        dims = []
        try:
            print('reading slice file...')
            with open(os.path.join(os.getcwd(), 'dims.txt'), 'r') as infile:
                content = infile.readlines()
                for line in content:
                    nums = [int(n) for n in line.split(' ')]
                    dims.append(nums)
        except Exception as e: 
            print(f'slice file reading failed, {e}...')
            return False
        return dims

