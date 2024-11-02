cmake_minimum_required(VERSION 3.5)
project(obstacle_detection)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pyrealsense2 REQUIRED)
find_package(ultralytics REQUIRED)
find_package(lapx REQUIRED)

install(PROGRAMS
  src/main_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()