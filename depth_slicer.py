from monodepth2.test_simple import create_depth_image_from_frame

from slice_grid_manager import SliceGridManager
from slice_grid import SliceGrid
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
    def __init__(self, method, init_frame, proportion_thresh, dist_thresh, slice_side_length=608, regen=True, square_size=100,
                grid_w=3, grid_h=4, refresh_rate=50): 
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
        if regen:
            self.generate_depth_image(init_frame)
            self.dims = self.calculate_dims()
        else: 
            self.dims = self.read_slice_file()
        
    
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
        prev_time = time.time()
        if self.method == 'simple':
            dims = self.calculate_simple_dims()
        elif self.method == 'precise': 
            dims = self.calculate_precise_dims()
        elif self.method == 'precise_grid':
            dims = self.calculate_precise_grid_dims()
        print('---------')
        print(f'Dimension calculation done in {time.time() - prev_time}')
        print('---------')
        # self.write_slice_file(dims)
        return dims

    
    def create_depth_mask(self, image, dim) -> bool:
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
                if dist < self.dist_thresh:
                    count += 1
        count = count / ((dim[3] - dim[2]) * (dim[1] - dim[0]))
        if count > self.proportion_thresh:
            return True
        return False
    

    def calculate_simple_dims(self) -> list:
        """
        Calculates "simple" method dimensions. Loops through every dimension to check for valid slices. Slice side length has biggest effect on outcome

        ...

        Parameters: 
        None
        """
        image = cv2.imread(self.path, cv2.IMREAD_COLOR)

        shape = image.shape
        height = int(shape[0])
        width = int(shape[1])

        # 608 is what darknet compresses images to
        num_slice_x = math.floor(width / self.slice_side_length)
        num_slice_y = math.floor(height / self.slice_side_length)

        slice_dim_x = math.ceil(width / num_slice_x)
        slice_dim_y = math.ceil(height / num_slice_y)

        no_comp_dims = []
        expand_amount = 0.05 * self.slice_side_length
        half_expand_amount = math.floor(expand_amount * .5)
        for x in range(1, num_slice_x + 1):
            for y in range(1, num_slice_y + 1):
                left = (x - 1) * slice_dim_x
                right = x * slice_dim_x
                top = (y - 1) * slice_dim_y
                bottom = y * slice_dim_y
                if bottom >= height:
                    bottom = height - 1
                    top = bottom - slice_dim_y
                if right >= width:
                    right = width - 1
                    left = right - slice_dim_x
                if top < 0:
                    top = 0
                if left < 0:
                    left = 0
                dim = [left, right, top, bottom]
                if self.create_depth_mask(image, dim):
                    no_comp_dims.append(dim)

        expand_amount = int(2 * half_expand_amount)

        # Bounds checking
        if len(no_comp_dims) > 1:
            for index, dim in enumerate(no_comp_dims):
                if dim[0] == 0:
                    dim[1] += expand_amount
                elif dim[1] >= width - 1:
                    dim[0] -= expand_amount
                else:
                    dim[0] -= half_expand_amount
                    dim[1] += half_expand_amount

                if dim[2] == 0:
                    dim[3] += expand_amount
                elif dim[3] >= height - 1:
                    dim[2] -= expand_amount
                else:
                    dim[2] -= half_expand_amount
                    dim[3] += half_expand_amount

                if dim[0] < 0:
                    dim[0] = 0
                if dim[1] >= width:
                    dim[1] = width-1
                if dim[2] < 0:
                    dim[2] = 0
                if dim[3] >= height:
                    dim[3] = height-1
        return no_comp_dims  
    

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

        sorted_blobs = [sorted(blob, key=lambda tup: tup[0]) for blob in blobs]
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

        processed_dims, _, _ = self.divide_island_boxes(dims, image_shape, self.slice_side_length)
        
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


    def divide_island_boxes(self, dims, image_shape, lower_bound=800) -> list: 
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
        for index, dim in enumerate(dims): 
            w = dim[1] - dim[0]
            h = dim[3] - dim[2]

            # if only one is less then just add it in 
            if w < lower_bound and h < lower_bound:
                processed_dims.append(dim)
                continue
    
            # want each stack to be ~800-1200 px 
            horiz_segments = int(math.floor(w / lower_bound))
            vert_segments = int(math.floor(h / lower_bound)) 
            if not horiz_segments: horiz_segments += 1
            if not vert_segments: vert_segments += 1
            horiz_remainder = w % lower_bound
            vert_remainder = h % lower_bound

            segment_width = lower_bound + horiz_remainder / horiz_segments if horiz_segments > 1 else w
            segment_height = lower_bound + vert_remainder / vert_segments if vert_segments > 1 else h

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
                    # check if both are good, then add half of overlap 
                    if new_dim[0] - woverlap > 0 and new_dim[1] + woverlap < image_shape[1] - 1:
                        new_dim[0] = new_dim[0] - woverlap
                        new_dim[1] = new_dim[1] + woverlap
                    elif new_dim[1] + 2 * woverlap < image_shape[1] - 1: 
                        new_dim[0] = 0
                        new_dim[1] = new_dim[1] + 2 * woverlap
                    elif new_dim[0] - 2 * woverlap > 0: 
                        new_dim[1] = image_shape[1] - 0
                        new_dim[0] = new_dim[0] - 2 * woverlap

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
                    graph.append((int(math.floor(y/square_size)), int(math.floor(x/square_size))))
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
                else: 
                    continue
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

            avg_xmax = 0
            avg_xmin = 0
            max_cols = 0
            for row in blob:
                if len(row) <= 1: continue
                if len(row) > max_cols: max_cols = len(row)
                avg_xmax += row[-1][1]
                avg_xmin += row[0][1]
            avg_xmax /= len(blob)
            avg_xmin /= len(blob)

            ymin_max = np.array([[100,0] for i in range(max_cols)])
            for row in blob:
                for coord in row:
                    #  if y is less than min 
                    if coord[0] < ymin_max[coord[1]][0]: ymin_max[coord[1]][0] = coord[0]
                    if coord[0] > ymin_max[coord[1]][1]: ymin_max[coord[1]][1] = coord[0]
            avg_ymin = sum(ymin_max[:, 0]) / len(ymin_max)
            avg_ymax = sum(ymin_max[:, 1]) / len(ymin_max)

            for y, row in enumerate(blob):
                row_cpy = row
                for i, coord in enumerate(row_cpy): 
                    if coord[1] < avg_xmin or coord[1] > avg_xmax: 
                        continue
                    if coord[0] < avg_ymin or coord[0] > avg_ymax:
                        continue
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
        with open(os.path.join(os.getcwd(), 'dims.txt'), 'r') as infile:
            content = infile.readlines()
            for line in content:
                nums = [int(n) for n in line.split(' ')]
                dims.append(nums)
        return dims

