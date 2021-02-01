from __future__ import division

import logging
import logging.config
import time
import math
import cv2
import numpy as np
import tensorflow as tf

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils
import os
import serial
import threading as t

# ser = serial.Serial( #Serial COM configuration
#     port='/dev/ttyACM0',
#     baudrate=19200,
#     # parity=serial.PARITY_NONE,
#     # stopbits=serial.STOPBITS_ONE,
#     # bytesize=serial.EIGHTBITS,
#     # rtscts=True,
#     timeout=1,
#   )
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=0.2)

strInput = "show ver"
ser.flush()
ser.write(strInput.encode('utf-8')+b'\n')


VIDEO_PATH = 'testdata/sample_video.mp4'
FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.1
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


FRAME_HEIGHT = 640
FRAME_WIDTH = 640
OFFSET = 200
DETECT_MIN = (int(FRAME_WIDTH/2)-OFFSET, 0)
DETECT_MAX = (int(FRAME_WIDTH/2)+OFFSET, FRAME_HEIGHT)


LEFT_START_POINT = (int(FRAME_WIDTH/2)-OFFSET, 0) 
LEFT_END_POINT = (int(FRAME_WIDTH/2)-OFFSET, FRAME_HEIGHT)

RIGHT_START_POINT = (int(FRAME_WIDTH/2)+OFFSET, 0) 
RIGHT_END_POINT = (int(FRAME_WIDTH/2)+OFFSET, FRAME_HEIGHT)
  
LINE_COLOR = (0, 0, 255) 

LINE_THICKNESS = 5

detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

# Read video from disk and count frames
cap = cv2.VideoCapture(0)

def distance(x1, x2, y1, y2):
    # width =math.sqrt( ((xmin-ymin)**2)+((xmax-ymin)**2) )
    # height = math.sqrt( ((xmax-ymin)**2)+((xmax-ymax)**2) )
    # area = int((width * height)/100)
    return math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )


SPEED = 10
DIRECTION= 30



def writeArduiono():
    while True:
        ACTION = (str(DIRECTION)+"#" +str(10)+ "\n").encode('utf_8')
        print(ACTION)
        ser.write(ACTION)
        line = ser.readline().decode('utf-8').rstrip()	
        print(line)

motorThread = t.Thread(target = writeArduiono, daemon=True)
motorThread.start()
hasStarted = False

with tf.Session(graph=detection_graph) as sess:
    while True:
        _, frame = cap.read()
        t = cv2.waitKey(20)
        if t == ord('q'):
            break
        
        crops, crops_coordinates = ops.extract_crops(
                frame, CROP_HEIGHT, CROP_WIDTH,
                CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)


        detection_dict = tf_utils.run_inference_for_batch(crops, sess)

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
        ymin, xmin, ymax, xmax = (0,0,0,0)

        detected = False
        hasLeft = False
        hasRight = False
        

        right_cone = None
        left_cone = None
        for box, score in zip(boxes, boxes_scores):
            if score > SCORE_THRESHOLD:
                ymin, xmin, ymax, xmax = box

                avg_x = (xmin+xmax)/2
                avg_y = (ymin+ymax)/2
                width = distance(xmin, xmax, ymin, ymax)
                height = distance(xmax, xmax, ymin, ymax)
                # height = math.sqrt( ((xmax-ymin)**2)+((xmax-ymax)**2) )
                area = int((width * height)/100)

                # if True:
                if(area > 10 and area < 500):

                    color_detected_rgb = cv_utils.predominant_rgb_color(
                        frame, ymin, xmin, ymax, xmax)
                    text = '{:.2f}'.format(score)
                    color_string = ""
                    if(color_detected_rgb.index(max(color_detected_rgb))) == 0:
                        color_string="BLUE"
                    elif (color_detected_rgb.index(max(color_detected_rgb))) == 1:
                        color_string="GREEN"
                    else:
                        color_string="RED"
                        
                    if((avg_x  > LEFT_START_POINT[0] and avg_x < RIGHT_START_POINT[0]) or (avg_y > LEFT_START_POINT[1] and avg_y < RIGHT_START_POINT[1]) ):
                        # LINE_COLOR = (0,0,255)
                        detected = True
                    # else:
                    #     LINE_COLOR = (0,255,0)
                    # print(avg_x, (FRAME_HEIGHT/2))
                    if(avg_x  < (FRAME_WIDTH/2)):
                        cone = "LEFT"
                    else:
                        cone = "RIGHT"
                    if(cone == "LEFT"):
                        if(hasLeft):
                            pass
                        else:
                            hasLeft = True
                            left_cone = ((xmax+xmax)/2, (ymin+ymax)/2)

                    if(cone == "RIGHT"):
                        if(hasRight):
                            pass
                        else:
                            hasRight = True
                            right_cone = ((xmax+xmax)/2, (ymin+ymax)/2)
                    # if(color_string== "BLUE" or color_string=="GREEN"):
                    cv_utils.add_rectangle_with_text(
                        frame, ymin, xmin, ymax, xmax,
                        color_detected_rgb, text + " " + color_string + " " + cone) 


        CENTER_X = (int(FRAME_WIDTH/2))

        if(detected):
            LINE_COLOR = (0,0,255)
        else:
            LINE_COLOR = (0,255,0)
        
        if left_cone is None:
            left_cone = (0,FRAME_HEIGHT/2)

        if right_cone is None:
            right_cone = (FRAME_WIDTH, FRAME_HEIGHT/2)
        
        _mid = (left_cone[0]+right_cone[0])/2
        if(detected):

            if((_mid < CENTER_X and _mid > LEFT_START_POINT[0])):
                DIRECTION = 60
            elif((_mid > CENTER_X and _mid < RIGHT_START_POINT[0])):
                DIRECTION = 0
        else:
            DIRECTION = 30

        if OUTPUT_WINDOW_WIDTH:
            frame = cv_utils.resize_width_keeping_aspect_ratio(
                frame, OUTPUT_WINDOW_WIDTH)
        cv2.circle(frame, (int(FRAME_WIDTH/2), int(FRAME_HEIGHT/2)), 20, (255,255,0), 2)
        cv2.line(frame, LEFT_START_POINT, LEFT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
        cv2.line(frame, RIGHT_START_POINT, RIGHT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
        cv2.imshow("f",frame)


cap.release()
cv2.destroyAllWindows()
