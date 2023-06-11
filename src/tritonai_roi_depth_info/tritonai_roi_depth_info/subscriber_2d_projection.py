import rclpy
from rclpy.node import Node

from std_msgs.msg import String
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
        super().__init__('minimal_subscriber')
        self.cones_cords = [[0], [0], [0]]
        self.boxes_cords = [[0], [0], [0]]

        self.subscription_box = self.create_subscription(
            String,
            '/box_xyz',
            self.box_callback,
            10)
        self.subscription_cone = self.create_subscription(
            String,
            '/cone_xyz',
            self.cones_callback,
            10)
        self.subscription_box  # prevent unused variable warning
        self.subscription_cone  # prevent unused variable warning
        # Path Planning
        self.planner = PathPlanner(MissionTypes.trackdrive)
        self.path_publisher = PahtPublisher()
        self.path_publisher_robohub = self.create_publisher(Path, '/trajectory', qos_profile_sensor_data)

        # self.fig = plt.figure()
        # self.ax1 = self.fig.add_subplot(1,1,1)
        # self.ani = animation.FuncAnimation(self.fig, self.animate_cones, interval=10)
        # plt.draw()

    def cones_callback(self, msg):
        data = json.loads(msg.data)
        emp = np.empty([0,2], dtype=np.float64)
        cones_position = [np.array([data['x_list'],data['z_list']]).transpose()/1000, emp ,emp, emp, emp]
        for i in range(len(data['x_list'])):
            self.cones_cords[0].append(data['x_list'][i])
            self.cones_cords[1].append(data['y_list'][i])
            self.cones_cords[2].append(data['z_list'][i])
        (
            path,
            sorted_left,
            sorted_right,
            left_cones_with_virtual,
            right_cones_with_virtual,
            left_to_right_match,
            right_to_left_match,
        ) = self.planner.calculate_path_in_global_frame(cones_position,np.array([0.000001,0.00001]),np.array([1.0,0.0]),return_intermediate_results=True)
        pub_msg = String()
        data['path'] = path.tolist()
        pub_msg.data = json.dumps(data)
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

        self.path_publisher_robohub.publish(path_msg)
        
        self.get_logger().info('Path is: "%s"' % path)
    
    def box_callback(self, msg):
        data = json.loads(msg.data)
        for i in range(len(data['x_list'])):
            self.boxes_cords[0].append(data['x_list'][i])
            self.boxes_cords[1].append(data['y_list'][i])
            self.boxes_cords[2].append(data['z_list'][i])
        self.get_logger().info('I heard: "%s"' % msg.data)


    # def animate_cones(self, i):
    #     cone_coords = np.array(self.cones_cords)
    #     self.ax1.clear()
    #     self.ax1.plot(cone_coords[0,:], cone_coords[2,:])


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