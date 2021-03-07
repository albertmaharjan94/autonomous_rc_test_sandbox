import cv2
import pyrealsense2 as rs
import numpy as np
import numpy as np



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_device('no')   
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile=pipeline.start(config)
# config.enable_record_to_file('object_detection.bag')
config.enable_record_to_file('intrinsics.bag')
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)
# Start streaming
e1 = cv2.getTickCount()
depth_stream = rs.video_stream()
depth_stream.type = rs.stream.depth
# print (type(depth_stream))
# depth_stream_w = rs.depth_intrinsics.width
# depth_stream_h = rs.intrinsics.height
depth_stream.fps = 30
depth_stream.bpp = 2
depth_stream.fmt = rs.format.z16
# depth_stream.intrinsics = intrinsics
depth_stream.index = 0
depth_stream.uid = 1312

# # 
# try:
#     while True:
#         profile = pipeline.get_active_profile()
#         print(profile)
#         depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
#         print(depth_profile)
#         depth_intrinsics = depth_profile.get_intrinsics()
#         print(depth_intrinsics)
#         w, h = depth_intrinsics.width, depth_intrinsics.height
#         print('width : ' ,w)
#         print('hight : ', h)
#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue
#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#         masked_a = np.ma.masked_equal(depth_image, 0.0, copy=False)
#         b=masked_a.max()
#         c=masked_a.min()

#         # depth = depth_image[320,240].astype(int)
#         # cv2.imshow('gray', depth)
#         # cv2.waitKey(1)
#         # # print('depth: ',depth)
#         distance = c * depth_scale
#         distance= format(distance , '0.3f')
#         # print ("Distance (m): ", distance)
#         print ("Distance (m): ", distance)
#         # print( depth_image[77,77])
# #      Apply colormap on depth image (image must be converted to 8_
#        # bit per pixel first)
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3),cv2.COLORMAP_JET)
#         # Stack both images horizontally
#         # cv2.imwrite(filename='pics/1.jpg', img=depth_colormap)
#         images = np.hstack((color_image, depth_colormap))
#         # Show images
#         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('RealSense', images)
#         cv2.waitKey(1)
# except Exception as e:
#     print(e)