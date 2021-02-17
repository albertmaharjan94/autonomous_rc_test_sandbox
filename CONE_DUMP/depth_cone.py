# import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import math

import os
import serial
import threading as t
import time

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils


ser = serial.Serial('/dev/ttyACM0', 19200, timeout=0.3)

strInput = "show ver"
ser.flush()
ser.write(strInput.encode('utf-8')+b'\n')



FRAME_WIDTH = 640
FRAME_HEIGHT = 480

OFFSET = 200

DETECT_MIN = (int(FRAME_WIDTH/2)-OFFSET, 0)
DETECT_MAX = (int(FRAME_WIDTH/2)+OFFSET, FRAME_HEIGHT)


LEFT_START_POINT = (int(FRAME_WIDTH/2)-OFFSET, 0) 
LEFT_END_POINT = (int(FRAME_WIDTH/2)-OFFSET, FRAME_HEIGHT)

RIGHT_START_POINT = (int(FRAME_WIDTH/2)+OFFSET, 0) 
RIGHT_END_POINT = (int(FRAME_WIDTH/2)+OFFSET, FRAME_HEIGHT)
  
LINE_COLOR = (0, 0, 255) 

LINE_THICKNESS = 5

SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


OVERRIDE = True
# Configure depth and color streams

# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)


print("[INFO] Starting streaming...")
# pipeline.start(config)
print("[INFO] Camera ready.")

# download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
print("[INFO] Loading model...")
PATH_TO_CKPT = "./models/ssd_mobilenet_v1/frozen_inference_graph.pb"

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

# coordinate distance
def distance(x1, x2, y1, y2):
    # width =math.sqrt( ((xmin-ymin)**2)+((xmax-ymin)**2) )
    # height = math.sqrt( ((xmax-ymin)**2)+((xmax-ymax)**2) )
    # area = int((width * height)/100)
    return math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )


SPEED = 0
DIRECTION= 30

detection_graph = tf_utils.load_model(PATH_TO_CKPT)

#  camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

hasStarted = False
# input to arduino 
def writeArduiono():
    while True:
        if hasStarted:
            if DIRECTION == 0 or DIRECTION == 90:
                ACTION = (str(DIRECTION)+"#" +str(SPEED)+ "\n").encode('utf_8')
                ser.write(ACTION)
                line = ser.readline().decode('utf-8').rstrip()	
                print(line)
                time.sleep(0.2)
            else:
                ACTION = (str(DIRECTION)+"#" +str(SPEED)+ "\n").encode('utf_8')
                ser.write(ACTION)
                line = ser.readline().decode('utf-8').rstrip()	
                print(line)
            print("Has started")

# start motor thread for individual process
motorThread = t.Thread(target = writeArduiono, daemon=True)
motorThread.start()

with tf.Session(graph=detection_graph) as sess:
    while True:
        hasStarted = True
        # frames = pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()

        # # Convert images to numpy arrays
        # color_image = np.asanyarray(color_frame.get_data())
        # scaled_size = (color_frame.width, color_frame.height)
        # # expand image dimensions to have shape: [1, None, None, 3]
        # # i.e. a single-column array, where each item in the column has the pixel RGB value
        # image_expanded = np.expand_dims(color_image, axis=0)
        # # Perform the actual detection by running the model with the image as input
        # (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
        #                                             feed_dict={image_tensor: image_expanded})



        _, frame = cap.read()
        
        crops, crops_coordinates = ops.extract_crops(
                        frame, FRAME_HEIGHT, FRAME_WIDTH,
                        FRAME_HEIGHT-20, FRAME_HEIGHT-20)
        
        detection_dict = tf_utils.run_inference_for_batch(crops, sess)

        # The detection boxes obtained are relative to each crop. Get
        # boxes relative to the original image
        # IMPORTANT! The boxes coordinates are in the following order:
        # (ymin, xmin, ymax, xmax)
        boxes = []
        for box_absolute, boxes_relative in zip(
                crops_coordinates, detection_dict['detection_boxes']):
            boxes.extend(ops.get_absolute_boxes(
                box_absolute,
                boxes_relative[np.any(boxes_relative, axis=1)]))
        if boxes:
            boxes = np.vstack(boxes)

        # Remove overlapping boxes
        boxes = ops.non_max_suppression_fast(
            boxes, NON_MAX_SUPPRESSION_THRESHOLD)

        # Get scores to display them on top of each detection
        boxes_scores = detection_dict['detection_scores']
        boxes_scores = boxes_scores[np.nonzero(boxes_scores)]
        detected = False
        hasLeft = False
        hasRight = False
        

        right_cone = None
        left_cone = None
        
        for box, score in zip(boxes, boxes_scores):
            if score > 0.01:
                left = int(box[1])
                top = int(box[0])
                right = int(box[3])
                bottom = int(box[2])

                # center of object
                avg_x = (left+right)/2
                avg_y = (top+bottom)/2

                # find the area of the object box
                width = distance(left, right, top, bottom)
                height = distance(left, right, top, bottom)                
                area = int((width * height)/100)


                # motor control only if area of the object is in between two values
                if(area > 300 and area < 1000):
                    p1 = (left, top)
                    p2 = (right, bottom)
                    r,g,b = cv_utils.predominant_rgb_color(
                            frame, top, left, bottom, right)
                    _color = None
                    if(g == 255):
                        _color ="GREEN"
                    else:
                        _color = "ORANGE"


                    if((avg_x  > LEFT_START_POINT[0] and avg_x < RIGHT_START_POINT[0]) 
                        or (avg_y > LEFT_START_POINT[1] and avg_y < RIGHT_START_POINT[1]) ):
                        detected = True
                    if(avg_x  < (FRAME_WIDTH/2)):
                        cone = "LEFT"
                    else:
                        cone = "RIGHT"

                    if(cone == "LEFT" and _color == "GREEN"):
                        if(hasLeft):
                            pass
                        else:
                            hasLeft = True
                            left_cone = (((right+right)/2, (top+bottom)/2),_color)

                            cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                            cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                                1, (b,g,r), 2, cv2.LINE_AA) 

                    if(cone == "RIGHT"  and _color == "ORANGE"):
                        if(hasRight):
                            pass
                        else:
                            hasRight = True
                            right_cone = (((right+right)/2, (top+bottom)/2), _color)

                            cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                            cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                                1, (b,g,r), 2, cv2.LINE_AA) 
                    
                    if(cone=="LEFT" and hasLeft == False and _color == "ORANGE"):
                        hasLeft = True
                        left_cone = (((right+right)/2, (top+bottom)/2),_color)

                        cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                        cv2.putText(frame, f"{r}, {g}, {b}", p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                            1, (b,g,r), 2, cv2.LINE_AA) 
                

            CENTER_X = (int(FRAME_WIDTH/2))

            
            if left_cone is None:
                left_cone = ((0,FRAME_HEIGHT/2), None)
                
            if right_cone is None:
                right_cone = ((FRAME_WIDTH, FRAME_HEIGHT/2), None)

            if left_cone[1] is not None and left_cone[1] =="ORANGE":
                if right_cone[1] is None or (right_cone[1] is not None and right_cone[1] =="ORANGE"):
                    right_cone = ((0, FRAME_HEIGHT/2), None)
                else:
                    left_cone = ((0,FRAME_HEIGHT/2), None)
            #  middle of two objects
            _mid = (left_cone[0][0]+right_cone[0][0])/2

            if(OVERRIDE == False):
                # if((_mid < CENTER_X and _mid > LEFT_START_POINT[0])):   
                #     DIRECTION = np.interp(_mid,[320,510],[30,60])
                #     # DIRECTION = 60
                # elif((_mid > CENTER_X and _mid < RIGHT_START_POINT[0]) or left_cone[1] == "ORANGE"):
                #     DIRECTION = np.interp(_mid,[125,320],[0,30])
                # else:
                #     DIRECTION = 30
                DIRECTION = int(np.interp(_mid,[125,510],[60,0]))
                SPEED = 10
            else:
                SPEED = 0
                DIRECTION = 30

        if(detected):
            LINE_COLOR = (0,0,255)
        else:
            LINE_COLOR = (0,255,0)
        # print(LINE_COLOR)
        cv2.circle(frame, (int(FRAME_WIDTH/2), int(FRAME_HEIGHT/2)), 20, (255,255,0), 2)
        cv2.line(frame, LEFT_START_POINT, LEFT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
        cv2.line(frame, RIGHT_START_POINT, RIGHT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        # debugging text
        cv2.putText(frame,f"Overide State: {OVERRIDE}", (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"Detected: {detected}", (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        
        cv2.putText(frame,f"Left Cone: {left_cone}", (10,int(FRAME_HEIGHT/2)-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"Right Cone: {right_cone}", (10,int(FRAME_HEIGHT/2)-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

        cv2.putText(frame,f"SPEED: {SPEED}", (FRAME_WIDTH-250, FRAME_HEIGHT-50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame,f"DIRECTION: {DIRECTION}", (FRAME_WIDTH-250,FRAME_HEIGHT-20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        try:
            cv2.putText(frame,f"MID: {_mid}", (FRAME_WIDTH-250,FRAME_HEIGHT-100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA) 
            cv2.line(frame, (int(_mid), 0), (int(_mid), FRAME_HEIGHT), (100,200,200), LINE_THICKNESS) 
        except:
            continue
        cv2.imshow('RealSense', frame)
        t = cv2.waitKey(1)
        if t == ord('q'):
            break
        
        if t == ord('s'):
            if OVERRIDE == False:
                OVERRIDE = True
                SPEED = 0
                print("Overide ON")
            else:
                OVERRIDE = False
                print("Overdie OFF")
print("[INFO] stop streaming ...")
# pipeline.stop()

