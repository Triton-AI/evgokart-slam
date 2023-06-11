import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Header
from custom_msgs.msg import ConesXYZ
from sensor_msgs.msg import PointCloud2
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


import numpy as np
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle, rotate
from fsd_path_planning.utils.cone_types import ConeTypes
from std_msgs.msg import Float32, Int32, Int32MultiArray

from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from .pointcloud2 import create_cloud_xyz32


class PahtPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(String, 'path', 10)

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.cones_cords = [[0], [0], [0]]
        self.boxes_cords = [[0], [0], [0]]

        self.cones_subscriber = self.create_subscription(
            ConesXYZ,
            "/lidar/cones",
            self.cones_callback,
            10
        )
        self.header_subscriber = self.create_subscription(
            PointCloud2,
            "/cone_points",
            self.header_callback,
            10
        )

        # Path Planning
        self.planner = PathPlanner(MissionTypes.trackdrive)
        self.path_publisher = PahtPublisher()
        self.path_publisher_robohub = self.create_publisher(Path, '/trajectory', qos_profile_sensor_data)
        self.path_publisher_pc2 = self.create_publisher(PointCloud2, '/traj_pc2', 10)
        CENTROID_TOPIC_NAME = '/centroid'
        self.centroid_error_publisher = self.create_publisher(Float32, CENTROID_TOPIC_NAME, 10)
        self.centroid_error_publisher
        self.centroid_error = Float32()
        self.header = Header()
        self.moving_list = []
        self.i = 0

    def header_callback(self, msg):
        self.header = msg.header

    def cones_callback(self,msg):
        cones_data = msg.cones
        cones_cords = [[],[],[]]
        emp = np.empty([0,2],dtype=np.float64)
        for cone in cones_data:
            cones_cords[0].append(cone.x)
            cones_cords[1].append(-cone.y)
            cones_cords[2].append(cone.z)
        
        cones_position = [np.array([cones_cords[1],cones_cords[0]]).transpose(), emp ,emp, emp, emp]

        (
            path,
            sorted_left,
            sorted_right,
            left_cones_with_virtual,
            right_cones_with_virtual,
            left_to_right_match,
            right_to_left_match,
        ) = self.planner.calculate_path_in_global_frame(cones_position,np.array([0.0,0.0]),np.array([0.0,1.0]),return_intermediate_results=True)
        pub_msg = String()
        msg = dict()
        # msg['heading_angle'] = heading_angle
        msg['cone_x'] = cones_cords[0]
        msg['cone_y'] = cones_cords[1]
        msg['cone_z'] = cones_cords[2]
        msg['path'] = path.tolist()
        pub_msg.data = json.dumps(msg)
        self.path_publisher.publisher_.publish(pub_msg)
        path_msg = Path()
        for p in path:
            pose_msg = PoseStamped()
            pose_msg.pose.position.x = p[1]
            pose_msg.pose.position.y = p[2]
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 0.0
            path_msg.poses.append(pose_msg)
        
        # path_points = []
        # for p in path:
        #     path_points.append((p[1],p[2],0))
        # pc2 = create_cloud_xyz32(self.header, path_points)
        # self.path_publisher_pc2.publish(pc2)
        self.path_publisher_robohub.publish(path_msg)
        if len(self.moving_list) < 5:
            self.moving_list.append(np.mean(path[:,1]))
        else:
            index = self.i%5
            self.moving_list[index] = np.mean(path[:,1])
        self.i = self.i + 1
        if self.i == 5:
            self.i = 0
        # moving_z_score(np.mean(path[:,1]),self.moving_list)
        self.centroid_error.data = float(np.mean(self.moving_list)/2.5)
        # self.centroid_error.data = float(np.mean(path[:,1])/2.2)
        self.centroid_error_publisher.publish(self.centroid_error)
        
        self.get_logger().info('Path is: "%s"' % path)

# def moving_z_score(x, arr, k=5, threshhold=2):
#     if(len(arr) < k):
#         arr.append(x)
#         return x
#     elif(len(arr) == k):
#         samp_mean = np.mean(arr)
#         samp_std = np.std(arr)
#         zscore = (x - samp_mean)/samp_std
#         if np.abs(zscore) < threshhold:
#             arr.append(x)
#             return x
#         else:
#             print("an outlier")
#     else:
#         samp_mean = np.mean(arr[len(arr) - k - 1:])
#         samp_std = np.std(arr[len(arr) - k - 1:])
#         zscore = (x-samp_mean)/samp_std
#         if(np.abs(zscore) < threshhold):
#             arr.append(x)
#             return x
#         else:
#             print("an outlier")

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()