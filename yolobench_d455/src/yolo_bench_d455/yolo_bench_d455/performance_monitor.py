#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time

class FPSMonitor(Node):
    def __init__(self):
        super().__init__('realsense_fps_monitor')
        
        # Declare parameters
        self.declare_parameter('topic', '/camera/camera/color/image_raw')
        
        # Get parameters
        self.topic = self.get_parameter('topic').value
        
        # Create subscription
        self.subscription = self.create_subscription(
            Image,
            self.topic,
            self.listener_callback,
            10)
        
        # Initialize counters
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        
        self.get_logger().info(f"Monitoring FPS on topic: {self.topic}")

    def listener_callback(self, msg):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.fps_history.append(fps)
            
            # Calculate average and max FPS
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            max_fps = max(self.fps_history)
            
            self.get_logger().info(f'Topic: {self.topic}')
            self.get_logger().info(f'Current FPS: {fps:.2f}')
            self.get_logger().info(f'Average FPS: {avg_fps:.2f}')
            self.get_logger().info(f'Maximum FPS: {max_fps:.2f}')
            self.get_logger().info('----------------------------')
            
            # Reset counters
            self.frame_count = 0
            self.start_time = current_time
            
            # Keep history limited
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)

def main(args=None):
    rclpy.init(args=args)
    fps_monitor = FPSMonitor()
    rclpy.spin(fps_monitor)
    fps_monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()