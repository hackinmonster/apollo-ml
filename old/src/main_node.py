import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
import cv2
import math
from std_msgs.msg import String

class Point:
    def __init__(self, cls, x, y):
        self.cls = cls
        self.x = x
        self.y = y
        self.depth = 0

    def toString(self):
        return f'{self.cls} at ({self.x},{self.y}): {float(self.depth):.3f}m'

class ObstacleDetectionNode(Node):
    def __init__(self):
        super().__init__('obstacle_detection_node')
        
        self.model = YOLO("yolov8n.pt")
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.publisher = self.create_publisher(String, 'detected_obstacles', 10)
        
        self.timer = self.create_timer(0.033, self.process_frames) 

    def process_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        results = self.model.track(color_image, verbose=False, persist=True, tracker="botsort.yaml")

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                if confidence < 0.6:
                    continue
                
                object_id = box.id.item()
                cls = self.classNames[int(box.cls[0])]

                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                depth = depth_frame.get_distance((x1 + x2) // 2, (y1 + y2) // 2)

                obstacle_info = f"Detected {cls} with ID {object_id} at depth {depth:.3f}m"
                self.publisher.publish(String(data=obstacle_info))
                self.get_logger().info(obstacle_info)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
