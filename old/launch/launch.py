from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='obstacle_detection',
            executable='main_node',
            name='obstacle_detection_node',
            output='screen'
        ),
    ])
