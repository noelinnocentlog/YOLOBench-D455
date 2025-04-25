#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolo_msgs.msg import DetectionArray  # Ensure this is the correct message type
from vision_msgs.msg import Detection2DArray 
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class DepthOverlayNode(Node):
    def __init__(self):
        super().__init__('depth_overlay_node')
        self.bridge = CvBridge()

        # Create subscribers using message_filters (for RGB, Depth, and Detections)
        self.rgb_sub = message_filters.Subscriber(self, Image, '/yolo/dbg_image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.dets_sub = message_filters.Subscriber(self, DetectionArray, '/yolo/detections')

        # Time synchronizer for syncing RGB, Depth, and Detection messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.dets_sub],
            queue_size=10,
            slop=0.5  # Adjusted slop to make synchronization more lenient
        )
        self.ts.registerCallback(self.callback)

        # Publisher for output image with depth info
        self.image_pub = self.create_publisher(Image, '/yolo/depth_dbg_image', 10)

        # Print initialization status
        self.get_logger().info("DepthOverlayNode initialized, waiting for data...")

    def callback(self, rgb_msg, depth_msg, dets_msg: DetectionArray):
        """
        Callback function to process RGB image, depth image, and detections.
        Adds depth information to the bounding boxes of the detections.
        """
        # Convert ROS2 messages to OpenCV images
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        print("Received image and depth data. Processing...")

        for det in dets_msg.detections:
            box = det.bbox

            # Handle Pose2D bounding box directly
            if hasattr(box, 'x') and hasattr(box, 'y') and hasattr(box, 'size_x') and hasattr(box, 'size_y'):
                # Extract bounding box details directly from Pose2D object
                x = int(box.x - box.size_x / 2)
                y = int(box.y - box.size_y / 2)
                w = int(box.size_x)
                h = int(box.size_y)
            else:
                # Fallback for other possible bounding box formats (if any)
                x = int(box.center.x - box.size_x / 2)
                y = int(box.center.y - box.size_y / 2)
                w = int(box.size_x)
                h = int(box.size_y)

            # Calculate the region of interest (ROI) in the depth image based on bounding box
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(x + w, depth_image.shape[1])
            y2 = min(y + h, depth_image.shape[0])

            # Extract depth values from the ROI
            depth_roi = depth_image[y1:y2, x1:x2]
            valid_depths = depth_roi[depth_roi > 0]

            if len(valid_depths) == 0:
                depth_m = 0.0  # No valid depth, set to 0
            else:
                # Median depth in meters (from mm to m)
                depth_m = np.median(valid_depths) / 1000.0

            # Draw bounding box and label with depth info on the RGB image
            label = f"{det.results[0].hypothesis.class_id} ({depth_m:.2f}m)"
            print(f"Detected: {det.results[0].hypothesis.class_id}, Depth: {depth_m:.2f} meters")

            # Draw bounding box
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
            # Add label (class_id and depth)
            cv2.putText(rgb_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Convert the OpenCV image back to a ROS2 message
        out_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
        out_msg.header = rgb_msg.header  # Maintain the original header

        # Publish the final image with depth information
        self.image_pub.publish(out_msg)
        print("Published the depth-enhanced image to /yolo/depth_dbg_image")

def main(args=None):
    """
    Main entry point for the ROS2 node. Initializes the node and starts processing.
    """
    rclpy.init(args=args)
    node = DepthOverlayNode()

    # Keep the node running until it's shutdown
    rclpy.spin(node)

    # Clean up when shutting down
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

