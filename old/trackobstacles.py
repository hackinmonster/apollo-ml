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
    def __init__(self, cls, x, y, depth):
        self.cls = cls
        self.x = x
        self.y = y
        self.depth = depth
    
    def toString(self):
        return f'{self.cls} at ({self.x},{self.y}): {float(self.depth):.3f}m'

#load model
model = YOLO("yolov8m.pt")

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
        points.append(Point('center', center_x, center_y, 0))

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

                    width = depth_frame.get_width()
                    height = depth_frame.get_height()
                    data = depth_frame.get_data()
                    depth_arr = np.asanyarray(data).reshape((height, width))

                    min_depth_value = np.percentile(depth_arr, 25)

                    min_depth_index = np.unravel_index(np.argmin(depth_arr, axis=None), depth_arr.shape)
                    min_depth_y, min_depth_x = min_depth_index

                    new_point = Point(cls, min_depth_x, min_depth_y, min_depth_value)

                    #print(str(classNames[int(box.cls[0])]) + "------>" + str(min_depth_value))
                    points.append(new_point)

                    if object_id not in object_info:

                        object_info[object_id] = {
                            'class': classNames[int(box.cls[0])],
                            'bbox': box.xyxy.tolist(),
                            'confidence': box.conf.item(),
                            'state': 'active',
                            'depths': []
                        }
                        object_info[object_id]['depths'].append(min_depth_value)
                    else:
                        object_info[object_id]['depths'].append(min_depth_value)

                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #exponentially weighted moving average

                #higher alpha means more weight to recent data (less smoothing)
                alpha = 0.35
                #higher r means lower sensitivity to change in direction
                r = 0.01

                if "curr_depth_avg" not in object_info[object_id]:
                    object_info[object_id]["curr_depth_avg"] = object_info[object_id]["depths"][-1]
                    object_info[object_id]["last_depth_avg"] = object_info[object_id]["depths"][-1]

                new_depth = object_info[object_id]["depths"][-1]
                last_depth_avg = object_info[object_id]["curr_depth_avg"]
                curr_depth_avg = alpha * new_depth + (1 - alpha) * last_depth_avg

                object_info[object_id]["last_depth_avg"] = last_depth_avg
                object_info[object_id]["curr_depth_avg"] = curr_depth_avg

                if curr_depth_avg > last_depth_avg * (1 + r):
                    print("FURTHER")
                elif curr_depth_avg < last_depth_avg * (1 - r):
                    print("CLOSER")
                else:
                    print("STOPPED")

                delta_depth = curr_depth_avg - last_depth_avg
                delta_time = 0.0333

                velocity = delta_depth / delta_time

                if velocity < 0:
                    TTC = curr_depth_avg / abs(velocity)
                else:
                    TTC = 1000

                if TTC < 5:
                    #stop robot
                    print("not implemented")
                
        
#todo: get box_width, box_height 
        '''
        for point in points:

            print("not implemented")

            
            box_x = point.x - box_width // 2
            box_y = point.y - box_height // 2

            text = "Depth: " + str(round(point.depth, 3))
            text2 = "TTC: " + str(round(TTC, 3))
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]
            text_x = box_x + (box_width - text_size[0]) // 2  # Center the text horizontally above the box
            text_y = box_y - 10  # Position the text just above the bounding box

            text_size2 = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]
            text_y2 = text_y + text_size[1] + 5  # Add space between the first and second lines of text

            cv2.putText(depth_colormap, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
            cv2.putText(depth_colormap, text2, (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
        '''

        cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth Stream', depth_colormap)
        key = cv2.waitKey(1)

        if key == 27:
            break

finally:
    pipeline.stop()