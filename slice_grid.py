import cv2
import numpy as np
import math 

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
    detections : 2D list of booleans 
        tracks which slices have detections 
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
        self.frame_count = 1
        self.full = False 
        self.max = w * h
        self.top = 0
        self.left = 0
        self.size = 0
        self.original_coord_grid = [[None for i in range(0, w)] for j in range(0, h)]
        self.slice_coord_grid = [[None for i in range(0, w)] for j in range(0, h)]
        self.detections = [[0 for i in range(0, w)] for j in range(0, h)]
        self.tiles = []

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
            self.tiles.append({
                'coord': (row, col),
                'grid_pixel_coord': (self.left, self.top),
                'orig_pixel_coord': dim,
                'detections': False
            })
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
        count = 0
        for row in self.original_coord_grid:
            width = 0
            max_tile_height = 0 
            for tile in row: 
                count += 1
                if count > self.size: break
                width += tile[1] - tile[0]
                if width > max_width: max_width = width
                if tile[3] - tile[2] > max_tile_height: max_tile_height = tile[3] - tile[2]
            if count > self.size: break
            height += max_tile_height
                
        self.image_width = max_width + 1 if max_width + 1 > self.width * self.tile_w else self.width * self.tile_w
        self.image_height = height + 1 if height + 1 > self.height * self.tile_h else self.height * self.tile_h 
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
        count = 0
        for y, row in enumerate(self.original_coord_grid): 
            for x, dim in enumerate(row): 
                count += 1
                if count > self.size: break
                tile_width = dim[1] - dim[0]
                tile_height = dim[3] - dim[2]
                ymin = self.slice_coord_grid[y][x][1]
                ymax = self.slice_coord_grid[y][x][1] + tile_height
                xmin = self.slice_coord_grid[y][x][0]
                xmax = self.slice_coord_grid[y][x][0] + tile_width
                self.frame_crop[ymin:ymax, xmin:xmax] = frame[dim[2]:dim[3], dim[0]:dim[1]]
            if count > self.size: break
        return self.frame_crop
    
    def update_grid(self, tiles): 
        for i, tile in enumerate(self.tiles):
            self.tiles[i] = {
                'coord': tile['coord'],
                'grid_pixel_coord': tile['grid_pixel_coord'], 
                'orig_pixel_coord': tiles[i]['orig_pixel_coord'],
                'detections': False
            }
            self.original_coord_grid[tile['coord'][0]][tile['coord'][1]] = tiles[i]['orig_pixel_coord']
  
    def normal_to_pixel_coords(self, coord):
        """
        Converts normalized bounding box coordinates to original image coordinates. 

        ...

        Parameters: 
        -----------
        coord : np.ndarray
            [left, top, right, bottom], 0.0 < 1.0
        """
        self.frame_count += 1

        left = coord[0] * self.image_width
        top = coord[1] * self.image_height
        right = coord[2] * self.image_width
        bottom = coord[3] * self.image_height

        mid_y = top + (bottom - top) / 2 
        mid_x = left + (right - left) / 2

        x = int(math.floor(mid_x / self.tile_w))
        y = int(math.floor(mid_y / self.tile_h))

        self.tiles[y*self.width+x]['detections'] = True

        left += self.original_coord_grid[y][x][0]
        right += self.original_coord_grid[y][x][0]
        top += self.original_coord_grid[y][x][2]
        bottom += self.original_coord_grid[y][x][2]

        self.detections[y][x] += 1

        return left - self.slice_coord_grid[y][x][0], right - self.slice_coord_grid[y][x][0], top - self.slice_coord_grid[y][x][1], bottom - self.slice_coord_grid[y][x][1]
