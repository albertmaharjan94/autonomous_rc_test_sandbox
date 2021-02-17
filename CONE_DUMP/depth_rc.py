# import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import math

import os
import serial
import threading as t
import time
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=0.2)

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

# input to arduino 
def writeArduiono():
    while True:
        if DIRECTION == 0 or DIRECTION == 90:
            ACTION = (str(DIRECTION)+"#" +str(SPEED)+ "\n").encode('utf_8')

            print(ACTION)
            ser.write(ACTION)
            line = ser.readline().decode('utf-8').rstrip()	
            print(line)
            time.sleep(0.2)
        else:
            ACTION = (str(DIRECTION)+"#" +str(SPEED)+ "\n").encode('utf_8')

            print(ACTION)
            ser.write(ACTION)
            line = ser.readline().decode('utf-8').rstrip()	
            print(line)


# start motor thread for individual process
motorThread = t.Thread(target = writeArduiono, daemon=True)
motorThread.start()
hasStarted = False


# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/
print("[INFO] Model loaded.")
colors_hash = {}
classes_90 = ["background", "person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
            "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
            "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ] 

#  camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
while True:
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
    
    # scaled_size = (frame.width, frame.height)
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image_expanded = np.expand_dims(frame, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                feed_dict={image_tensor: image_expanded})
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)
       
    detected = False
    hasLeft = False
    hasRight = False
    

    right_cone = None
    left_cone = None
    
    for idx in range(int(num)):
        class_ = classes[idx]
        score = scores[idx]
        box = boxes[idx]

        # print(class_,classes_90[class_])

        if class_ not in colors_hash:
            colors_hash[class_] = tuple(np.random.choice(range(256), size=3))
        
	# 47 is cup class
        if score > 0:
        # if score > 0.01 and class_ == 47:
            left = int(box[1] * FRAME_WIDTH)
            top = int(box[0] * FRAME_HEIGHT)
            right = int(box[3] * FRAME_WIDTH)
            bottom = int(box[2] * FRAME_HEIGHT)

            avg_x = (left+right)/2
            avg_y = (top+bottom)/2
            width = distance(left, right, top, bottom)
            height = distance(left, right, top, bottom)
			# height = math.sqrt( ((xmax-ymin)**2)+((xmax-ymax)**2) )
            area = int((width * height)/100)
            # print(area)
            if(area > 45 and area < 1000):
                p1 = (left, top)
                p2 = (right, bottom)
		        # draw box
                r, g, b = colors_hash[class_]

                if((avg_x  > LEFT_START_POINT[0] and avg_x < RIGHT_START_POINT[0]) or (avg_y > LEFT_START_POINT[1] and avg_y < RIGHT_START_POINT[1]) ):
                    detected = True
                if(avg_x  < (FRAME_WIDTH/2)):
                    cone = "LEFT"
                else:
                    cone = "RIGHT"
                if(cone == "LEFT"):
                    if(hasLeft):
                        pass
                    else:
                        hasLeft = True
                        left_cone = ((right+right)/2, (top+bottom)/2)

                        cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                        cv2.putText(frame, classes_90[class_], p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                            1, (255,0,0), 2, cv2.LINE_AA) 

                if(cone == "RIGHT"):
                    if(hasRight):
                        pass
                    else:
                        hasRight = True
                        right_cone = ((right+right)/2, (top+bottom)/2)

                        cv2.rectangle(frame, p1, p2, (int(r), int(g), int(b)), 2, 1)
                        cv2.putText(frame, classes_90[class_], p1,  cv2.FONT_HERSHEY_SIMPLEX,  
                            1, (255,0,0), 2, cv2.LINE_AA) 
            

        CENTER_X = (int(FRAME_WIDTH/2))

        if(detected):
            LINE_COLOR = (0,0,255)
        else:
            LINE_COLOR = (0,255,0)
        
        if left_cone is None:
            left_cone = (0,FRAME_HEIGHT/2)

        if right_cone is None:
            right_cone = (FRAME_WIDTH, FRAME_HEIGHT/2)
        
        #  middle of two objects
        _mid = (left_cone[0]+right_cone[0])/2
        if(OVERRIDE == False):
            if(detected):
                if((_mid < CENTER_X and _mid > LEFT_START_POINT[0])):   
                    
                    DIRECTION = 60
                elif((_mid > CENTER_X and _mid < RIGHT_START_POINT[0])):
                    DIRECTION = 0
            else:
                DIRECTION = 30
            SPEED = 10
        else:
            SPEED = 0
            DIRECTION = 30
    cv2.circle(frame, (int(FRAME_WIDTH/2), int(FRAME_HEIGHT/2)), 20, (255,255,0), 2)
    cv2.line(frame, LEFT_START_POINT, LEFT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
    cv2.line(frame, RIGHT_START_POINT, RIGHT_END_POINT, LINE_COLOR, LINE_THICKNESS) 
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.putText(frame,f"Overide State: {OVERRIDE}", (10,30),  cv2.FONT_HERSHEY_SIMPLEX,  
		               1, (255,0,0), 2, cv2.LINE_AA) 

    cv2.imshow('RealSense', frame)
    t = cv2.waitKey(1)
    if t == ord('q'):
        break
    
    if t == ord('s'):
        if OVERRIDE == False:
            OVERRIDE = True
            print("Overide ON")
        else:
            OVERRIDE = False
            print("Overdie OFF")
print("[INFO] stop streaming ...")
# pipeline.stop()

