from setuptools import setup

package_name = 'tritonai_roi_depth_info'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='triton-ai',
    maintainer_email='triton-ai@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = tritonai_roi_depth_info.object_detection_publisher:main',
            'listener = tritonai_roi_depth_info.subscriber_2d_projection:main',
            'path_listener = tritonai_roi_depth_info.path_display_lidar:main',
            'listener_lidar = tritonai_roi_depth_info.subscriber_2d_projection_lidar:main'
        ],
    },
)
