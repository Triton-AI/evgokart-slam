# Triton AI: evGoKart SLAM

## Overview
This is the code repository for our UCSD CSE 145/237D Spring 2023 Project developed with the Triton AI team led by Prof. Jack Silberman for the Autonomous Karting Series race which happens every year at Purdue University.

The goal of this project an autonomous evGoKart on a track without a previous known map using Simultaneous Localization and Mapping (SLAM).

## Team
The members working on this project as part of the course are:
1. Aashish Bhole, Grad Student, Computer Science and Engineering
2. Aleksandra Desens, Undergrad Student, Computer Engineering

The members from Triton AI who helped us are:
1. Moses Wong, Undergrad Student, Mathematics
2. Janette Lin, Undergrad Student, Data Science
3. Gordon Zhao, Grad Student, Robotics

## Approach
The race track lanes are marked by cones and there maybe some obstacles along the path. We need to enable real time localization and mapping to race reliably. For navigation between the cones, we worked on implementing object detection using a camera and LiDAR and then map an optimal path for the gokart to follow.
We struggled to fuse the camera and LiDAR data together using the code snippets provided by the Livox LiDAR repositories. Also, due to unreliable camera depth information due to misalignment of the RGB and Stereo depth frames because of the fisheye camera model, we used just the LiDAR for cone clustering and path planning purposes.

## Process Flow
Insert image

## Project Structure
```
evgokart-slam
├── racetrack_lidar_rosbag        <-- contains the LiDAR rosbag of the race track for simulation
├── README.md                     <-- README file for this repository
├── requirements.txt              <-- python packages required to run
└── src
    ├── custom_msgs               <-- required by the lidar_clustering package
    ├── lidar_clustering          <-- ros2 package for cone clustering using LiDAR pointcloud data
    ├── livox_lidar_pkg           <-- ros2 package for the Livox LiDARs
    ├── livox_ros_driver2         <-- ros2 drivers for the Livox LiDARs
    └── tritonai_roi_depth_info   <-- ros2 package for path planning and visualization
```

## Installation
```
git clone <this-repository>
cd evgokart-slam
pip install -r requirements.txt
```

### Setup ROS2 environment and build package
Need ROS2 pre-installed. We used the ROS-Foxy version for our development.
```
source /opt/ros/foxy/setup.bash
colcon build
source install/setup.bash
```

### IMPORTANT Must move fsd_path_planning into ros2 site-packages for it to work
Requires python3.9+ ideally but we have modified the code to work with python3.8as our ROS2 environment was created using it.
```
cp -r src/tritonai_roi_depth_info/fsd_path_planning install/tritonai_roi_depth_info/lib/python3.X/site-packages/
```


### Replay rosbag and launch LiDAR visualization:
In case of unavailability of LiDAR sensor, you can run the rosbag provided to simulate point cloud data of the race track. The LiDAR model we used was Livox HAP, you will need to install the respective ros2 drivers of your LiDAR sensor and run the visualization accordingly.
```
ros2 bag play race_track_lidar_rosbag/
ros2 launch livox_ros_driver2 rviz_HAP_launch.py
```

### Cone Clustering and Path Planning:
Launch the ROS2 nodes for clustering the LiDAR pointcloud data to identify cones marking the lanes of the track and create a race line to follow using the path planning algorithm mentioned before (fsd_path_planning) built by Formula Student Team TU Berlin.
```
ros2 run tritonai_roi_depth_info listener_lidar
ros2 run lidar_clustering clustering
```

### Path Visualization
To see the position of cones and the race line created, launch the following ROS2 node:
```
ros2 run tritonai_roi_depth_info path_listener
```

## Project Demo
Insert video link

