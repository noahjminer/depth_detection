import cv2
import numpy as np

img = cv2.imread('frame_disp.jpeg',0)
real_img = cv2.imread('testdims.jpeg')

_, mask = cv2.threshold(img, thresh=60, maxval=20, type=cv2.THRESH_BINARY_INV)
mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3 channel mask
im_thresh_gray = cv2.bitwise_and(real_img, real_img, mask=mask)

cv2.imwrite('mask.jpg', im_thresh_gray)
