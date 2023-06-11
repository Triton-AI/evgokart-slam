import rclpy
from rclpy.node import Node
import numpy as np
# import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
from .pointcloud2 import *
import json



class lidar_data_listener(Node):

    def __init__(self):
        # call super() in the constructor to initialize the Node object
        # the parameter we pass is the node name
        super().__init__('lidar_data_subscriber')

        # create a timer sending two parameters:
        # - the duration between two callbacks (0.2 seconds)
        # - the timer function (timer_callback)
        # self.create_timer(50, self.listener_callback)

        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        
    def listener_callback(self, msg):
        # xyz = np.array([[0,0,0]])
        # gen = read_points(msg, skip_nans=True)
        # for p in list(gen):
        #     xyz = np.append(xyz,[[p[0],p[1],p[2]]], axis = 0)

        # out_pcd = o3d.geometry.PointCloud()    
        # out_pcd.points = o3d.utility.Vector3dVector(xyz)
        # o3d.io.write_point_cloud("/home/projects/tmp/cloud.ply",out_pcd)

        f = open("pointcloud_data.txt", "w")

        cloud_points_coordinates = []

        for point in read_points(msg, field_names=('x','y','z'), skip_nans= True):
            cloud_points_coordinates.append(point)

        
        json.dump(cloud_points_coordinates, f)

        # self.get_logger().info('I heard: "%s"' % cloud_points_coordinates)

        f.close

def main(args=None):
    # initialize the ROS2 communication
    rclpy.init(args=args)
    # declare the node constructor
    node = lidar_data_listener()
    # keeps the node alive, waits for a request to kill the node (ctrl+c)
    rclpy.spin(node)
    node.destroy_node()
    # shutdown the ROS2 communication
    rclpy.shutdown()

if __name__ == '__main__':
    main()
