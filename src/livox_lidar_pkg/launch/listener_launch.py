from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='livox_lidar_pkg',
            executable='lidar_listener',
            output='screen'),
    ])
