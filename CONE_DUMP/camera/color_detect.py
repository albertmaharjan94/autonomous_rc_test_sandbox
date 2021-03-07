import cv2 

from dominant_color_detection import detect_colors
from PIL import Image, ImageColor
import numexpr as ne
import numpy as np
def bincount_numexpr_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    eval_params = {'a0':a2D[:,0],'a1':a2D[:,1],'a2':a2D[:,2],
                   's0':col_range[0],'s1':col_range[1]}
    a1D = ne.evaluate('a0*s0*s1+a1*s0+a2',eval_params)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

# Open image using openCV2 
opencv_image = cv2.imread("ora.jpg") 
_bin = bincount_numexpr_app(opencv_image)
print(_bin)
# Displaying the Scanned Image by using cv2.imshow() method 
cv2.imshow("OpenCV Image", opencv_image) 
  
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 