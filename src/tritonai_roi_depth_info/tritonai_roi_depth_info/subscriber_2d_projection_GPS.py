import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


import numpy as np
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle, rotate
from fsd_path_planning.utils.cone_types import ConeTypes

from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class PahtPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(String, 'path', 10)

class MinimalSubscriber(Node):

    def __init__(self):
        self.x_pose_prev = 0.000001
        self.y_pose_prev = 0.000001
        POSE_TOPIC_NAME = '/fix'
        super().__init__('minimal_subscriber')
        self.cones_cords = [[0], [0], [0]]
        self.boxes_cords = [[0], [0], [0]]

        self.subscription_box = self.create_subscription(
            String,
            '/box_xyz',
            self.box_callback,
            10)
        self.subscription_box  # prevent unused variable warning

        self.subscription_cone = self.create_subscription(
            String,
            '/cone_xyz',
            self.cones_callback,
            10)
        self.subscription_cone  # prevent unused variable warning

        self.pose_subscriber = self.create_subscription(
            NavSatFix, 
            POSE_TOPIC_NAME, 
            self.set_pose, 
            10
            )
        self.pose_subscriber

        # Path Planning
        self.planner = PathPlanner(MissionTypes.trackdrive)
        self.path_publisher = PahtPublisher()
        self.path_publisher_robohub = self.create_publisher(Path, '/trajectory', qos_profile_sensor_data)

        # Cone Filtering
        self.known_cones = np.empty([0,2],dtype=np.float64) # Initializing empty array
        self.epsilon = 2


    # Call back function for getting GPS x,y
    def set_pose(self,pose_data):
        # if not self.initial_pose_set:
        self.x_pose_prev=pose_data.longitude
        self.y_pose_prev=pose_data.latitude
        self.theta_pose_prev=0.0
        # self.initial_pose_set=True


    def cones_callback(self, msg):
        data = json.loads(msg.data)
        emp = np.empty([0,2], dtype=np.float64)
        del_yaw = -data['heading_angle']
        # Rotation Matrix
        R = np.array([[np.cos(del_yaw), -np.sin(del_yaw)], [np.sin(del_yaw), np.cos(del_yaw)]])
        car_direction = np.array([0,1])@R
        car_direction = car_direction/np.linalg.norm(car_direction)
        # Cone poistion rotation, then affine shift
        new_cones = np.array(list(zip(data['x_list'],data['z_list'])))@R/1000 + np.array([self.x_pose_prev,self.y_pose_prev])
        self.add_new_cones(new_cones) # Add new cones to list of known cones
        cones_position = [self.known_cones, emp ,emp, emp, emp]
        # cones_position = [np.array([data['x_list'],data['z_list']]).transpose()/1000, emp ,emp, emp, emp]
        print("loading")
        (
            path,
            sorted_left,
            sorted_right,
            left_cones_with_virtual,
            right_cones_with_virtual,
            left_to_right_match,
            right_to_left_match,
        ) = self.planner.calculate_path_in_global_frame(cones_position,np.array([self.x_pose_prev,self.y_pose_prev]),car_direction,return_intermediate_results=True)
        pub_msg = String()
        data['car_direction'] = car_direction.tolist()
        print("Done!")
        data['path'] = path.tolist()
        data['new_cone_x'] = self.known_cones[:,0].tolist()
        data['new_cone_z'] = self.known_cones[:,1].tolist()
        pub_msg.data = json.dumps(data)
        self.path_publisher.publisher_.publish(pub_msg)
        # Publishing a path way point for Dom's way pt follow
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

        self.path_publisher_robohub.publish(path_msg)
        
        # self.get_logger().info('Path is: "%s"' % path)

    def add_new_cones(self,new_cones):
        for detection in new_cones:
            if not (np.linalg.norm(self.known_cones-detection,axis = 1) < self.epsilon).any():
                self.known_cones = np.append(self.known_cones,np.array([detection]),axis = 0)
                # print(self.known_cones)


    
    def box_callback(self, msg):
        data = json.loads(msg.data)
        for i in range(len(data['x_list'])):
            self.boxes_cords[0].append(data['x_list'][i])
            self.boxes_cords[1].append(data['y_list'][i])
            self.boxes_cords[2].append(data['z_list'][i])
        self.get_logger().info('I heard: "%s"' % msg.data)


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