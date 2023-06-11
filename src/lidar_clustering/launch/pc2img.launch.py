from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_clustering',
            executable='clustering',
            name='clustering',
            remappings=[
                ('/livox/lidar', '/livox/lidar'),
                ('lidar/colored_clusters', 'lidar/colored_clusters'),
            ]
        ),
        Node(
            package='lidar_clustering',
            executable='pc2img',
            name='pc2img',
            remappings=[
                ('lidar/data', '/lidar/colored_clusters'),
                ('lidar/image', '/lidar/image'),
            ]
        )
    ])