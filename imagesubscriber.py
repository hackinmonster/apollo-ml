#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

def depth_callback(data):
    rospy.loginfo("Received depth image")

def color_callback(data):
    rospy.loginfo("Received color image")

def main():
    rospy.init_node('image_subscriber_node', anonymous=True)

    # Subscribe to depth and color image topics
    rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback)
    rospy.Subscriber('/camera/color/image_raw', Image, color_callback)

    rospy.spin()  # Keep the node running

if __name__ == '__main__':
    main()
