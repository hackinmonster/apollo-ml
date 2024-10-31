#the basic idea:
    #avoid obstacles by classifying them and using depth camera 

    #draw bounding box using classification model (yolo)
    #find center point of this box
    #calculate distance and time to collision (TTC)
    #do this for all present objects + center point
    #use TTC to inform robot behavior (break/slow down)

#useful links:
#https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples
#https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.depth_frame.html#pyrealsense2.depth_frame

#python 3.11.10

import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import cv2
import math

#represent a pixel/point on screen
class Point:
    def __init__(self, cls, x, y):
        self.cls = cls
        self.x = x
        self.y = y
        self.depth = 0
    
    def toString(self):
        return f'{self.cls} at ({self.x},{self.y}): {float(self.depth):.3f}m'

#load model
model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]


#realsense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


previous_depth = None
velocity = 0  # m/s
TTC = float('inf')  # initialized as big val
object_info = {}


try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        #used for display
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        points = []

        #center point (can be adjusted to any point)
        #we need this to detect of theres a barrier directly in front of the robot
        center_x = depth_colormap.shape[1] // 2
        center_y = depth_colormap.shape[0] // 2
        points.append(Point('center', center_x, center_y))

        #use rgb color image to make model predictions
        results = model.track(color_image, verbose=False, persist=True, tracker="botsort.yaml")

        for result in results:
                boxes = result.boxes

                threshold = 0.6

                #print(object_info)

                #iterate through each of the boxes in that frame
                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    if confidence < threshold:
                        continue

                    object_id = box.id.item() if box.id is not None else None

                    cls = classNames[int(box.cls[0])]

                    #print(box.id.item())

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    h, w, _ = color_image.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    new_x = (x1 + x2) // 2
                    new_y = (y1 + y2) // 2
                    new_point = Point(cls, new_x, new_y)
                    points.append(new_point)

                    depth = depth_frame.get_distance(new_x, new_y)

                    if object_id not in object_info:

                        object_info[object_id] = {
                            'class': classNames[int(box.cls[0])],
                            'bbox': box.xyxy.tolist(),
                            'confidence': box.conf.item(),
                            'state': 'active',
                            'depths': []
                        }
                        object_info[object_id]['depths'].append(depth)
                    else:
                        object_info[object_id]['depths'].append(depth)

                    #add rectangle to our window
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    #if box.id is not None:
                        #print(cls + " -> " + str(box.id.item()))

                    if len(object_info[object_id]["depths"]) > 1:
                        curr_depth = object_info[object_id]["depths"][-1]
                        last_depth = object_info[object_id]["depths"][-2]

                        #print(curr_depth)
                        if curr_depth > last_depth * 1.02:
                            print("FURTHER")
                        elif curr_depth < last_depth * .98:
                            print("CLOSER")


        for point in points:
            point.depth = depth_frame.get_distance(point.x, point.y)
            #print(point.toString())


        for point in points:
            x,y = point.x,point.y
            
        #the end output of this module is to identify if something is getting closer quickly

        #to do this, we need to be able to calculate time-to-collision for each of the objects classified, plus the center point
        #in order to do that, we have to compare a point's current depth to its last depth (delta depth)   

        #we cant track by pixel because the center point of an object can change
        #instead we should track by object
        #since there can be multiple objects of the same class (like people), we need unique identifiers for each              

        #we also need to consider which objects are a priority (moving vs stationary, different types?)

        #and finally we need to determine a threshold for what constitutes an obstacle that should be avoided, and how (complete stop or deaccelerate?)

        #old code, some might be useful:

        '''
        if len(object_info) > 1:




            previous_depth = object_info[-2]




            #change in distance
            delta_depth = current_depth - previous_depth
            
            #fps (we need change in time to measure velocity)
            delta_time = 0.0167  # seconds

            velocity = delta_depth / delta_time
            
            if velocity < 0:  #depth decreasing
                #time to collision
                TTC = current_depth / abs(velocity)
            else:
                TTC = 0

        '''
        
        
        #adding each point to our window for visualization
        for point in points:

            #there will be a circle, then two lines of text above it, one for depth, one for TTC
            cv2.circle(depth_colormap, (point.x, point.y), radius=10, color=(0, 0, 0), thickness=-1)
            text = "Depth: " + str(round(point.depth, 3))
            #text2 = "TTC: " + str(round(TTC, 3))
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, thickness=20)[0]
            text_x = point.x - text_size[0] // 2  
            text_y = point.y - 30 
            text_y2 = point.y - 15

            cv2.putText(depth_colormap, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
            #cv2.putText(depth_colormap, text2, (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)


        #show window
        cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth Stream', depth_colormap)
        key = cv2.waitKey(1)

        #esc key
        if key == 27:
            break

finally:
    pipeline.stop()