## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

pipeline_back = rs.pipeline()
config_back = rs.config()
config_back.enable_device('035422073295')
config_back.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
config_back.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
config_back.enable_stream(rs.stream.infrared,1, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
config_back.enable_stream(rs.stream.infrared,2, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)

# Start streaming
# pipeline.start(config)
pipeline_back.start(config_back)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline_back.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = color_image.copy()
        images[20:220,20:220,:] = cv2.resize(cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV), (200, 200))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()