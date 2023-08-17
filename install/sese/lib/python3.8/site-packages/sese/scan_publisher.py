import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import LaserScan
from vision_msgs.msg import ObjectHypothesisWithPose


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('blabla')
        self.publisher_ = self.create_publisher(LaserScan, 'scan', 10)
        self.subsciber = self.create_subscription(LaserScan, 'my_scan', self.subscriber_callback, 10)
        self.rosbag_subscriber = self.create_subscription(ObjectHypothesisWithPose, 'Camera', self.position_callback, 10)
        timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.resolution = 180
        self.angle_min = -1.570000
        self.angle_max = 1.570000

        self.x = 0.
        self.y = 0.
        self.x_previous = 0.
        self.y_previous = 0.

        self.counter = 0
        self.distance = 5

        self.label = ''

    def position_callback(self, msg):
        if msg.pose.pose.position.z == self.x and msg.pose.pose.position.x == self.y:
            self.counter += 1
        elif msg.pose.pose.position.z == 0 and msg.pose.pose.position.x == 0:
            self.counter += 1
        else:
            self.counter = 0
        print(self.counter)


        if msg.pose.pose.position.z > 0.5 and msg.pose.pose.position.z < 20:
            self.x = float(msg.pose.pose.position.z) / 3
        else:
            self.x = 0.0
            self.y = 0.0
        if msg.pose.pose.position.x >= 0.1 or msg.pose.pose.position.x <= -0.1:
            self.y = float(msg.pose.pose.position.x) / 3
        else:
            self.x = 0.0
            self.y = 0.0
        self.label = msg.id


        print(f'Odczytana pozycja x: {msg.pose.pose.position.z}')
        print(f'Odczytana pozycja y: {msg.pose.pose.position.x}')
        print(f'Zapisana pozycja x: {self.x}')
        print(f'Zapisana pozycja y: {self.y}')

    def subscriber_callback(self, msg):
        # self.y = -0.00000000001
        # self.x = self.distance
        # self.distance -= 0.00001
        if self.x == self.x_previous and self.y == self.y_previous and self.counter >= 3:
            self.x = 0.
            self.y = 0.
            self.counter = 0
        elif self.x == self.x_previous and self.y == self.y_previous:
            self.counter += 1
        elif self.x != self.x_previous and self.y != self.y_previous:
            self.counter = 0
        # if self.y == self.y_previous:
        #     self.y = 0.
        # initialize the result
        result = 0

        # set the radius of circle on which detected point is located
        R = np.sqrt(pow(self.x, 2) + pow(self.y, 2))

        # set the length of arc for every section of circle
        total_range = (self.angle_max - self.angle_min) * R
        section = total_range / self.resolution

        # set length of arc between starting point of circle's arc and detected point
        if self.y == 0 or self.x == 0:
            R = 0
            arc = 0
        elif self.y >= 0:
            alfa = np.arctan(self.y/self.x)
            arc = alfa * R + (int(self.resolution/2)) * section
        else:
            alfa = np.arctan(np.abs(self.y)/self.x)
            arc = (int(self.resolution/2))*section-alfa * R

        # get the section in which the detected point is in. Thats the result
        for i in range(self.resolution):
            sekcja = section*(i+1)
            # print(sekcja)
            if  sekcja < arc:
                result += 1
            else:
                break

        for i in range(result-3,result+3):
            msg.ranges[i] = float(R)
        # msg.ranges[result] = float(R)
        # self.get_logger().info(f'ranges: {msg.ranges}')
        self.publisher_.publish(msg)
        self.i += 1
        
        self.x_previous = self.x
        self.y_previous = self.y
        print('label:', self.label)
        print(f'x: {self.x}')
        print(f'y: {self.y}')
        print(f'x_previous: {self.x_previous}')
        print(f'y_previous: {self.y_previous}')
        print(f'counter: {self.counter}')
        # self.distance -= 0.1

    def timer_callback(self):
        msg = LaserScan()
        msg.header.frame_id = 'Dzien dobry'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.angle_min = 0.0
        msg.angle_max = np.pi
        msg.angle_increment = (msg.angle_max - msg.angle_min) / self.resolution
        msg.time_increment = 0.001
        msg.scan_time = 0.5
        msg.range_min = 0.5
        msg.range_max = 20.
        msg.ranges = [1.]*self.resolution

        for i in range(100, 120):
            msg.ranges[i] = 2.

        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.distance -= 0.1
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()