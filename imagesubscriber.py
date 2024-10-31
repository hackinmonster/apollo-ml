#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageSubscriberNode(Node):
    def __init__(self):
        super().__init__('image_subscriber_node')

        # Subscribe to depth and color image topics
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)

    def depth_callback(self, data):
        self.get_logger().info("Received depth image")

    def color_callback(self, data):
        self.get_logger().info("Received color image")

def main(args=None):
    rclpy.init(args=args)

    image_subscriber_node = ImageSubscriberNode()

    rclpy.spin(image_subscriber_node)  # Keep the node running

    image_subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
