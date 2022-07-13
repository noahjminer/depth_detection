# Depth Detection 

This project utilizes depth estimation to predict objects in high resolution images more accurately. While it is a stand alone package, this repo provides a pipeline for testing on videos. 

To run this project, install the required packages in requirements.txt through pip, then run python video.py

`python video.py --path=path/to/video.mp4 --method=mask|precise|precise_grid`

This project uses Python 3.8

## Args guide

--path (string) **required**

*specifies path to video. Result is saved in same directory as video*

--depth_thresh (float)

*0.0 to 1.0, with 0.0 being farthest. Default .2. Threshold for how far a pixel is to be selected in slicing*

--prop_thresh (float)

*0.0 to 1.0, default .9, proportion of pixels that are below depth_thresh for slice to be elligible*

--method (string)

*chooses method of detection. default and most refined is mask, other options are precise and precise_grid*

--slice_side_length (int)

*default 800. specifies minimum side length of slices*

--square_size (int)

*default 50. Divides initial frame into squares of this size. Used for testing, not really important, 50 is a good number*

--grid_width (int)

*default 3. Used in precise_grid to create the grid slice images. Does not impact other methods*

--grid_height (int)

*default 4. Same as grid_width*

--refresh_rate (int)

*default 100. Used in precise_grid to determine how often grid slices are refreshed*

# Methods

## Calibration

Upon generating a greyscale depth estimation image, key areas are then determined by pixel value. This is where the different methods come in. 

The "mask" method uses opencv's threshold function to create a mask of pixels under a threshold (currently needs to be connected to monodepth2). 

The "precise" and "precise_grid" method just uses the depth estimated image.

The image is divided in squares. The mask method checks for values in the masked image above 0. Precise and precise_grid checks every pixel with RGB / 255 less than the depth_threshold. If the count of elligible pixels / total pixels is above t he prop_thresh, the square is added to a list.

The list is then grouped by neighboring squares, giving back islands of squares across the image. Using the min and max of their x and y coordinates, the final dimensions are calculated. Then there is some overlap adjustments and some padding in the case that a dimension is too small. 



