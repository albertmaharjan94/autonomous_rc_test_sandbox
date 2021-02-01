from __future__ import division

import logging
import logging.config
import time

import cv2
import numpy as np
import tensorflow as tf

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils
import os


VIDEO_PATH = 'testdata/sample_video.mp4'
FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


FRAME_HEIGHT = 640
FRAME_WIDTH = 640
OFFSET = 100

LEFT_START_POINT = (int(FRAME_WIDTH/2)-OFFSET, 0) 
LEFT_END_POINT = (int(FRAME_WIDTH/2)-OFFSET, FRAME_HEIGHT)

RIGHT_START_POINT = (int(FRAME_WIDTH/2)+OFFSET, 0) 
RIGHT_END_POINT = (int(FRAME_WIDTH/2)+OFFSET, FRAME_HEIGHT)
  
LINE_COLOR = (0, 0, 255) 

LINE_THICKNESS = 5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

# Read video from disk and count frames
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

with tf.Session(graph=detection_graph) as sess:
    while cap.isOpened():

        if DETECT_EVERY_N_SECONDS:
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    processed_images * fps * DETECT_EVERY_N_SECONDS)

        ret, frame = cap.read()
        if ret:
            # crops are images as ndarrays of shape
            # (number_crops, CROP_HEIGHT, CROP_WIDTH, 3)
            # crop coordinates are the ymin, xmin, ymax, xmax coordinates in
            #  the original image
            print(frame)
            crops, crops_coordinates = ops.extract_crops(
                frame, CROP_HEIGHT, CROP_WIDTH,
                CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)

            # Uncomment this if you also uncommented the two lines before
            #  creating the TF session.
            # crops = np.array([crops[0]])
            # crops_coordinates = [crops_coordinates[0]]

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

            for box, score in zip(boxes, boxes_scores):
                if score > SCORE_THRESHOLD:
                    ymin, xmin, ymax, xmax = box
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
                    if(color_string== "BLUE" or color_string=="GREEN"):
                        cv_utils.add_rectangle_with_text(
                            frame, ymin, xmin, ymax, xmax,
                            color_detected_rgb, text + " " + color_string)
                    

            if OUTPUT_WINDOW_WIDTH:
                frame = cv_utils.resize_width_keeping_aspect_ratio(
                    frame, OUTPUT_WINDOW_WIDTH)
            cv2.line(frame, LEFT_START_POINT, LEFT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
            cv2.line(frame, RIGHT_START_POINT, RIGHT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
    
    
            cv2.imshow('Detection result', frame)
            t = cv2.waitKey(20)
            time.sleep(0.5)
            if t == ord('q'):
                break

        else:
            # No more frames. Break the loop
            break

cap.release()
cv2.destroyAllWindows()