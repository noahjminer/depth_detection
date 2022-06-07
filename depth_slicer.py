from monodepth2.test_simple import create_depth_image_from_frame

import cv2 
import numpy as np
import math
import os 
import time

class SliceGrid: 
    """
    A class for storing a grid of dimensions cut from a frame

    Attributes: 
    width : int
        How many columns in grid
    height : int 
        How many rows in grid
    tile_w : int 
        width of each tile in pixels 
    tile_h : int 
        height of each tile in pixels 
    image_width : int 
        width of grid in pixels
    image_height : int 
        height of grid in pixels
    frame_crop : numpy array (image_width, image_height, 3)
        stores current frame's grid
    full : bool 
        False when size < max 
    max : int 
        amount of grid squares, w*h
    top : int 
        used to track pixel value of each row 
    left : int 
        used to track pixel value of each column 
    size : int 
        amount of grid squares already filled
    original_coord_grid : 2D list of dims [xmin, xmax, ymin, ymax] (in original image coordinates)
        stores grid square location in original frame
    slice_coord_grid : 2D list of (left, top) in pixels
        stores grid square location in grid image
    """
    def __init__(self, tile_w, tile_h, w=4, h=3):
        """
        Parameters: 

        tile_w : int 
            tile width in pixels
        tile_h : int 
            tile height in pixels 
        w : int 
            amount of columns 
        h : int 
            amount of rows
        """
        self.width = w 
        self.height = h
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.image_width = 0
        self.image_height = 0
        self.frame_crop = None
        self.full = False 
        self.max = w * h
        self.top = 0
        self.left = 0
        self.size = 0
        self.original_coord_grid = [[None for i in range(0, w)] for j in range(0, h)]
        self.slice_coord_grid = [[None for i in range(0, w)] for j in range(0, h)]

    def insert_tile(self, dim) -> None:
        """
        Inserts a tile into next available slot of grid 

        ...

        Parameters: 
        -----------
        dim : List of [xmin, xmax, ymin, ymax, blob_index]
            Values stored in original_coord_grid for later use
        """
        dim_w = dim[1] - dim[0]
        dim_h = dim[3] - dim[2]
        if not self.full:
            if self.size: 
                row = int(math.floor(self.size / self.width))
                col = self.size % self.width
            else: 
                row = 0
                col = 0
            
            self.original_coord_grid[row][col] = dim
            self.slice_coord_grid[row][col] = [self.left, self.top]
            if col == self.width - 1:
                self.left = 0 
                self.top += dim_h
            else: 
                self.left += dim_w
            self.size += 1
            if self.size >= self.max: self.full = True
    
    def create_empty_image(self) -> None:
        """
        Creates an empty image of image_width * image_height pixels
        """
        width = 0
        height = 0
        max_width = 0
        max_tile_height = 0
        for row in self.original_coord_grid:
            width = 0
            max_tile_height = 0 
            for tile in row: 
                width += tile[1] - tile[0]
                if width > max_width: max_width = width
                if tile[3] - tile[2] > max_tile_height: max_tile_height = tile[3] - tile[2]
            height += max_tile_height
                
        self.image_width = max_width + 1 
        self.image_height = height + 1
        self.frame_crop = np.zeros((self.image_height, self.image_width, 3), np.uint8)
        
    def make_grid_image(self, frame) -> np.ndarray:
        """
        Crops frame with slice_coord_grid and original_coord_grid dimensions

        ...

        Parameters: 
        -----------
        frame : np.ndarray
            current frame in video
        """ 
        for y, row in enumerate(self.original_coord_grid): 
            for x, dim in enumerate(row): 
                tile_width = dim[1] - dim[0]
                tile_height = dim[3] - dim[2]
                ymin = self.slice_coord_grid[y][x][1]
                ymax = self.slice_coord_grid[y][x][1] + tile_height
                xmin = self.slice_coord_grid[y][x][0]
                xmax = self.slice_coord_grid[y][x][0] + tile_width
                self.frame_crop[ymin:ymax, xmin:xmax] = frame[dim[2]:dim[3], dim[0]:dim[1]]
        return self.frame_crop
    
    def normal_to_pixel_coords(self, coord):
        """
        Converts normalized bounding box coordinates to original image coordinates. 

        ...

        Parameters: 
        -----------
        coord : np.ndarray
            [left, top, right, bottom], 0.0 < 1.0
        """
        left = coord[0] * self.image_width
        top = coord[1] * self.image_height
        right = coord[2] * self.image_width
        bottom = coord[3] * self.image_height

        mid_y = top + (bottom - top) / 2 
        mid_x = left + (right - left) / 2

        for i in range(0, len(self.slice_coord_grid)):
            if i + 1 < len(self.slice_coord_grid):
                if mid_y < self.slice_coord_grid[i+1][0][1]: 
                    y = i 
                    break
            else: y = i
        
        for i in range(0, len(self.slice_coord_grid[y])):
            if i + 1 < len(self.slice_coord_grid[y]):
                if mid_x < self.slice_coord_grid[y][i+1][0]: 
                    x = i
                    break
            else: x = i

        left += self.original_coord_grid[y][x][0]
        right += self.original_coord_grid[y][x][0]
        top += self.original_coord_grid[y][x][2]
        bottom += self.original_coord_grid[y][x][2]

        return left - self.slice_coord_grid[y][x][0], right - self.slice_coord_grid[y][x][0], top - self.slice_coord_grid[y][x][1], bottom - self.slice_coord_grid[y][x][1]



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
    def __init__(self, method, init_frame, proportion_thresh, dist_thresh, slice_side_length=608, regen=True, square_size=100): 
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

        avgs = []
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
        
        # Process connected blobs into one box
        dims = self.create_bounding_box_for_islands(blobs, image_shape, self.square_size)
        
        # Process dims by splitting up bigger boxes 
        return self.divide_island_boxes(dims, image_shape, self.slice_side_length)


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
        processed_dims = self.divide_island_boxes(dims, image_shape, 250)
        
        tile_width = 300 
        tile_height = 400

        slice = SliceGrid(tile_width, tile_height)
        slices = [slice]
        for dim in processed_dims: 
            if slice.full: 
                slice.create_empty_image()
                slice = SliceGrid(tile_width,tile_height)
                slices.append(slice)
            slice.insert_tile(dim)
        slice.create_empty_image()
        
        return slices


    def divide_island_boxes(self, dims, image_shape, lower_bound): 
        processed_dims = [] 
        for index, dim in enumerate(dims): 
            w = dim[1] - dim[0]
            h = dim[3] - dim[2]

            # if only one is less then we can stack
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

            woverlap = 0.05 * segment_width
            hoverlap = 0.05 * segment_height
            
            for i in range(1, horiz_segments+1): 
                for j in range(1, vert_segments+1): 
                    new_dim = [0,0,0,0]
                    new_dim[0] = dim[0] + (i-1) * segment_width
                    new_dim[1] = dim[0] + (i-1) * segment_width + segment_width
                    new_dim[2] = dim[2] + (j-1) * segment_height
                    new_dim[3] = dim[2] + (j-1) * segment_height + segment_height
                    new_dim[0] = new_dim[0] - woverlap if new_dim[0] - woverlap > 0 else 0
                    new_dim[1] = new_dim[1] + woverlap if new_dim[1] + woverlap < image_shape[1] - 1 else image_shape[1] - 1
                    new_dim[2] = new_dim[2] - hoverlap if new_dim[2] - hoverlap > 0 else 0
                    new_dim[3] = new_dim[3] + hoverlap if new_dim[3] + hoverlap < image_shape[0] - 1 else image_shape[0] - 1
                    new_dim = [int(math.floor(num)) for num in new_dim]
                    new_dim.append(index)
                    processed_dims.append(new_dim)

        return processed_dims

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
                blob.append(neighbor)
                try:
                    if graph.count(neighbor) > 0:
                        graph.remove(neighbor)
                except ValueError as e: 
                    print(e)
                    print(neighbor)
                    break
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
            for coord in blob: 
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

