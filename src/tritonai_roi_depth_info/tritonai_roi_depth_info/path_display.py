import rclpy
import json
from rclpy.node import Node
import numpy as np

from std_msgs.msg import String
import matplotlib.pyplot as plt

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/path',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        car_position = [0.000001,0.000001]
        data = json.loads(msg.data)
        car_direction = np.array(data['car_direction'])
        x_pos = np.array(data['new_cone_x'])
        z_pos = np.array(data['new_cone_z'])
        path = np.array(data['path'])
        plt.scatter(x_pos, z_pos, c='k')
        plt.plot(*path[:, 1:3].T)
        plt.plot(
        [car_position[0], car_position[0] + car_direction[0]],
        [car_position[1], car_position[1] + car_direction[1]],
        c="k",
        )
        plt.axis("equal")
        plt.show()
        # self.get_logger().info('I heard: "%s"' % msg.data)


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