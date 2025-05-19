#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import tf2_ros
from geometry_msgs.msg import PointStamped, TransformStamped
import tf2_geometry_msgs

class DepthOverlayNode(Node):
    def __init__(self):
        super().__init__('depth_overlay_node')
        self.bridge = CvBridge()
        
        # Parameters for depth filtering
        self.declare_parameter('min_depth', 0.1)  # Min valid depth in meters
        self.declare_parameter('max_depth', 10.0)  # Max valid depth in meters
        self.declare_parameter('depth_method', 'median')  # Options: 'median', 'mean', 'min'
        self.declare_parameter('roi_scale', 0.5)  # Scale factor for ROI relative to bounding box
        
        # Get parameters
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.depth_method = self.get_parameter('depth_method').value
        self.roi_scale = self.get_parameter('roi_scale').value
        
        # Create subscribers using message_filters
        self.rgb_sub = message_filters.Subscriber(self, Image, '/yolo/dbg_image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        self.dets_sub = message_filters.Subscriber(self, DetectionArray, '/yolo/detections')
        
        # Optional: Subscribe to camera info for accurate depth calculation
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/depth/camera_info',
            self.camera_info_callback,
            10)
        self.camera_intrinsics = None
        
        # Time synchronizer for syncing RGB, Depth, and Detection messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.dets_sub],
            queue_size=10,
            slop=0.5  # Adjusted slop to make synchronization more lenient
        )
        self.ts.registerCallback(self.callback)
        
        # Publisher for output image with depth info
        self.image_pub = self.create_publisher(Image, '/yolo/depth_dbg_image', 10)
        
        # Initialize TF2 listener for coordinate transforms if needed
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Print initialization status
        self.get_logger().info("DepthOverlayNode initialized with parameters:")
        self.get_logger().info(f"- min_depth: {self.min_depth}m")
        self.get_logger().info(f"- max_depth: {self.max_depth}m")
        self.get_logger().info(f"- depth_method: {self.depth_method}")
        self.get_logger().info(f"- roi_scale: {self.roi_scale}")
        self.get_logger().info("Waiting for data...")

    def camera_info_callback(self, msg):
        """Store camera intrinsics for potential distance corrections"""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],  # Focal length x
                'fy': msg.k[4],  # Focal length y
                'cx': msg.k[2],  # Principal point x
                'cy': msg.k[5],  # Principal point y
                'width': msg.width,
                'height': msg.height
            }
            self.get_logger().info("Camera intrinsics received")

    def filter_depth_values(self, depth_values):
        """
        Filter out invalid or outlier depth values.
        
        Args:
            depth_values: NumPy array of depth values
            
        Returns:
            Filtered depth values
        """
        # Convert from mm to m if needed (D455 typically outputs in mm)
        if np.median(depth_values) > 1000:  # Simple heuristic to check if in mm
            depth_values = depth_values / 1000.0
            
        # Remove zeros (invalid measurements)
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) == 0:
            return np.array([])
            
        # Filter out values outside reasonable range
        valid_depths = valid_depths[(valid_depths >= self.min_depth) & 
                                   (valid_depths <= self.max_depth)]
        
        # Optional: Statistical outlier removal
        if len(valid_depths) > 10:
            q1 = np.percentile(valid_depths, 25)
            q3 = np.percentile(valid_depths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            valid_depths = valid_depths[(valid_depths >= lower_bound) & 
                                       (valid_depths <= upper_bound)]
        
        return valid_depths

    def calculate_object_depth(self, depth_image, x, y, w, h):
        """
        Calculate the object's depth using a scaled ROI within the bounding box.
        
        Args:
            depth_image: The depth image
            x, y, w, h: Bounding box coordinates
            
        Returns:
            Depth in meters and confidence value
        """
        # Ensure coordinates are within image bounds
        img_h, img_w = depth_image.shape
        
        # Calculate center of bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate scaled ROI dimensions (focus on center of object)
        roi_w = int(w * self.roi_scale)
        roi_h = int(h * self.roi_scale)
        
        # Calculate ROI coordinates
        roi_x1 = max(0, center_x - roi_w // 2)
        roi_y1 = max(0, center_y - roi_h // 2)
        roi_x2 = min(img_w, roi_x1 + roi_w)
        roi_y2 = min(img_h, roi_y1 + roi_h)
        
        # Extract ROI from depth image
        depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Filter depth values
        valid_depths = self.filter_depth_values(depth_roi.flatten())
        
        if len(valid_depths) == 0:
            return 0.0, 0.0  # No valid depth, zero confidence
        
        # Calculate depth based on selected method
        confidence = len(valid_depths) / (roi_w * roi_h)  # Ratio of valid pixels
        
        if self.depth_method == 'median':
            depth_m = np.median(valid_depths)
        elif self.depth_method == 'mean':
            depth_m = np.mean(valid_depths)
        elif self.depth_method == 'min':
            # Minimum non-zero depth (closest point)
            depth_m = np.min(valid_depths)
        else:
            # Default to median
            depth_m = np.median(valid_depths)
            
        return depth_m, confidence

    def callback(self, rgb_msg, depth_msg, dets_msg: DetectionArray):
        """
        Callback function to process RGB image, depth image, and detections.
        Adds depth information to the bounding boxes of the detections.
        """
        try:
            # Convert ROS2 messages to OpenCV images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            self.get_logger().info("Received image and depth data. Processing...")
            
            # Debug info about depth image
            depth_stats = {
                'shape': depth_image.shape,
                'dtype': depth_image.dtype,
                'min': np.min(depth_image),
                'max': np.max(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 0,
                'mean': np.mean(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 0
            }
            
            self.get_logger().debug(f"Depth image stats: {depth_stats}")
            
            # Check if depth and RGB have different resolutions and adjust if needed
            if depth_image.shape[:2] != rgb_image.shape[:2]:
                self.get_logger().warn(f"Depth ({depth_image.shape[:2]}) and RGB ({rgb_image.shape[:2]}) resolutions don't match!")
                # Resize depth to match RGB if needed
                depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            
            # Loop over all detections
            for det in dets_msg.detections:
                box = det.bbox
                
                # Extract bounding box details
                center_x = int(box.center.position.x)
                center_y = int(box.center.position.y)
                width = int(box.size.x)
                height = int(box.size.y)
                
                # Compute top-left corner coordinates of bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                w = int(width)
                h = int(height)
                
                # Calculate depth using the improved method
                depth_m, confidence = self.calculate_object_depth(depth_image, x, y, w, h)
                
                # Skip if no valid depth
                if depth_m <= 0:
                    label = f"{det.class_name} (no depth)"
                    self.get_logger().warn(f"No valid depth for {det.class_name}")
                else:
                    # Format label with depth and confidence
                    label = f"{det.class_name} ({depth_m:.2f}m, conf:{confidence:.2f})"
                    self.get_logger().info(f"Detected: {det.class_name}, Depth: {depth_m:.2f}m, Confidence: {confidence:.2f}")
                
                # Draw bounding box with different colors based on depth
                if depth_m <= 0:
                    color = (0, 0, 255)  # red for invalid
                elif confidence < 0.3:
                    color = (0, 165, 255)  # orange for low confidence
                else:
                    # Color gradient from green (close) to blue (far)
                    green = max(0, min(255, int(255 * (1 - depth_m / self.max_depth))))
                    blue = max(0, min(255, int(255 * depth_m / self.max_depth)))
                    color = (blue, green, 0)
                
                # Draw bounding box
                cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)
                
                # Add background to text for better visibility
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(rgb_image, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
                cv2.putText(rgb_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Optional: Draw the ROI used for depth calculation to debug
                roi_w = int(w * self.roi_scale)
                roi_h = int(h * self.roi_scale)
                roi_x = x + w//2 - roi_w//2
                roi_y = y + h//2 - roi_h//2
                cv2.rectangle(rgb_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 255), 1)
            
            # Add depth method info to the image
            info_text = f"Depth method: {self.depth_method}, ROI scale: {self.roi_scale}"
            cv2.putText(rgb_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert the OpenCV image back to a ROS2 message
            out_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
            out_msg.header = rgb_msg.header  # Maintain the original header
            
            # Publish the final image with depth information
            self.image_pub.publish(out_msg)
            self.get_logger().info("Published the depth-enhanced image to /yolo/depth_dbg_image")
            
        except Exception as e:
            self.get_logger().error(f"Error processing data: {str(e)}")

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
