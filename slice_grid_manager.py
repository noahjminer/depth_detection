from operator import itemgetter
import time

class SliceGridManager: 
    """
    A class for managing SliceGrids and adjusting them based on detection rate. 

    ... 

    Attributes: 
    -----------
    grids : list of SliceGrids 
        Holds the current slice grids 
    frame_count : int 
        frame count of current video
    detection_threshold : float 
        the minimum detections / frames ratio for a slice to be made inactive
    """

    def __init__(self, grids, detection_threshold=0.05, refresh_interval=10):
        self.grids = grids
        self.active_grid_indices = []
        self.front_grid_index = 0
        self.detection_threshold = detection_threshold
        self.refresh_interval = refresh_interval
    
    def get_slices(self, frame, all=False): 
        slices = []
        prev = time.time()
        if all: 
            self.active_grid_indices.clear()
            for index, grid in enumerate(self.grids): 
                self.active_grid_indices.append(index)
                slices.append(grid.make_grid_image(frame))
        else:
            for index in self.active_grid_indices: 
                slices.append(self.grids[index].make_grid_image(frame))
        print(f'crop took {time.time()-prev}')
        return slices
    
    def check_detection(self):
        prev = time.time()
        if not len(self.grids): return 
        all_tiles = []
        for grid in self.grids: all_tiles += grid.tiles
        sorted_tiles = sorted(all_tiles, key=itemgetter('detections'), reverse=True)
        lower_bound = 0
        self.active_grid_indices.clear()
        for i, grid in enumerate(self.grids): 
            grid.update_grid(sorted_tiles[lower_bound:lower_bound+len(grid.tiles)])
            if any([tile['detections'] for tile in sorted_tiles[lower_bound:lower_bound+len(grid.tiles)]]):
                self.active_grid_indices.append(i)
            lower_bound += len(grid.tiles)
        print(f'check_detections took {time.time()-prev}')
                

