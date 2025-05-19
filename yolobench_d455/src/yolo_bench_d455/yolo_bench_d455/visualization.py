#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import time
import threading
from geometry_msgs.msg import Point

class Visualization(Node):
    def __init__(self):
        super().__init__('visualization')
        
        # Declare parameters
        self.declare_parameter('tracking_topic', '/yolo/depth_tracking_image')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('publish_point_cloud', True)
        self.declare_parameter('point_cloud_downsample', 4)  # Downsample factor for point cloud
        self.declare_parameter('visualization_rate', 10.0)   # Hz for visualization updates
        self.declare_parameter('colormap', 'jet')            # OpenCV colormap for depth visualization
        
        # Get parameters
        self.tracking_topic = self.get_parameter('tracking_topic').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.publish_point_cloud = self.get_parameter('publish_point_cloud').value
        self.point_cloud_downsample = self.get_parameter('point_cloud_downsample').value
        self.visualization_rate = self.get_parameter('visualization_rate').value
        self.colormap = self.get_parameter('colormap').value
        
        # Initialize bridge
        self.bridge = CvBridge()
        
        # Latest data storage
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_tracking = None
        self.lock = threading.RLock()
        
        # Set up QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers
        self.tracking_sub = self.create_subscription(
            Image, self.tracking_topic, self.tracking_callback, 10)
        
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, sensor_qos)
        
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sensor_qos)
        
        # Publishers
        self.depth_colormap_pub = self.create_publisher(
            Image, '/visualization/depth_colormap', 10)
        
        self.side_by_side_pub = self.create_publisher(
            Image, '/visualization/side_by_side', 10)
        
        if self.publish_point_cloud:
            self.point_cloud_pub = self.create_publisher(
                PointCloud2, '/visualization/point_cloud', 10)
            
        self.overlay_pub = self.create_publisher(
            Image, '/visualization/tracking_overlay', 10)
        
        # Timer for visualization
        self.timer = self.create_timer(
            1.0/self.visualization_rate, self.visualization_timer_callback)
        
        # Set up colormaps
        self.colormaps = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'bone': cv2.COLORMAP_BONE,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'cividis': cv2.COLORMAP_CIVIDIS,
        }
        
        # Log initialization
        self.get_logger().info(f"Visualization node initialized with parameters:")
        self.get_logger().info(f"- Tracking topic: {self.tracking_topic}")
        self.get_logger().info(f"- Publish point cloud: {self.publish_point_cloud}")
        self.get_logger().info(f"- Colormap: {self.colormap}")
        
    def tracking_callback(self, msg):
        try:
            with self.lock:
                self.latest_tracking = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error in tracking callback: {str(e)}")
    
    def rgb_callback(self, msg):
        try:
            with self.lock:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error in RGB callback: {str(e)}")
    
    def depth_callback(self, msg):
        try:
            with self.lock:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error in depth callback: {str(e)}")
    
    def visualization_timer_callback(self):
        try:
            with self.lock:
                # Check if we have all the data
                if self.latest_rgb is None or self.latest_depth is None:
                    return
                
                # Make copies to avoid threading issues
                rgb_copy = self.latest_rgb.copy()
                depth_copy = self.latest_depth.copy()
                tracking_copy = None if self.latest_tracking is None else self.latest_tracking.copy()
            
            # Resize depth to match RGB if needed
            if depth_copy.shape[:2] != rgb_copy.shape[:2]:
                depth_copy = cv2.resize(depth_copy, (rgb_copy.shape[1], rgb_copy.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Create and publish depth colormap
            depth_colormap = self.create_depth_colormap(depth_copy)
            self.depth_colormap_pub.publish(
                self.bridge.cv2_to_imgmsg(depth_colormap, encoding='bgr8'))
            
            # Create and publish side-by-side view
            side_by_side = self.create_side_by_side_view(rgb_copy, depth_colormap, tracking_copy)
            self.side_by_side_pub.publish(
                self.bridge.cv2_to_imgmsg(side_by_side, encoding='bgr8'))
            
            # Create and publish tracking overlay
            if tracking_copy is not None:
                # If we have tracking data, create a blend of RGB and tracking
                overlay = self.create_tracking_overlay(rgb_copy, tracking_copy)
                self.overlay_pub.publish(
                    self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8'))
            
            # Create and publish point cloud if enabled
            if self.publish_point_cloud and tracking_copy is not None:
                point_cloud_msg = self.create_point_cloud(rgb_copy, depth_copy)
                self.point_cloud_pub.publish(point_cloud_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error in visualization timer: {str(e)}")
    
    def create_depth_colormap(self, depth_image):
        """Create a colormap visualization of depth data."""
        # Normalize depth to 0-255 range
        min_depth = 0.1  # 10cm
        max_depth = 10.0  # 10m
        
        # Convert from mm to m if needed
        if np.median(depth_image[depth_image > 0]) > 1000:
            depth_image_m = depth_image / 1000.0
        else:
            depth_image_m = depth_image.copy()
        
        # Clip depth to valid range
        depth_image_m = np.clip(depth_image_m, min_depth, max_depth)
        
        # Create mask for valid depth
        valid_mask = depth_image_m > 0
        
        # Normalize to 0-255 for visualization
        norm_depth = np.zeros_like(depth_image_m, dtype=np.uint8)
        if np.any(valid_mask):
            # Use log scale for better visualization
            norm_depth[valid_mask] = 255 * (1.0 - np.log(depth_image_m[valid_mask]) / np.log(max_depth))
        
        # Apply colormap
        colormap_id = self.colormaps.get(self.colormap, cv2.COLORMAP_JET)
        depth_colormap = cv2.applyColorMap(norm_depth, colormap_id)
        
        # Set invalid regions to black
        depth_colormap[~valid_mask] = [0, 0, 0]
        
        return depth_colormap
    
    def create_side_by_side_view(self, rgb_image, depth_colormap, tracking_image=None):
        """Create a side-by-side view of RGB, depth, and tracking images."""
        # Ensure all images are the same size
        h, w = rgb_image.shape[:2]
        depth_colormap_resized = cv2.resize(depth_colormap, (w, h))
        
        # If we have tracking data, create a 3-panel view
        if tracking_image is not None:
            tracking_resized = cv2.resize(tracking_image, (w, h))
            # Create a 1x3 grid
            side_by_side = np.hstack((rgb_image, depth_colormap_resized, tracking_resized))
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            font_color = (255, 255, 255)
            
            # Calculate text positions
            label_y = 30
            rgb_label_x = w // 2 - 50
            depth_label_x = w + w // 2 - 50
            tracking_label_x = 2 * w + w // 2 - 50
            
            # Add text
            cv2.putText(side_by_side, "RGB", (rgb_label_x, label_y), 
                       font, font_scale, font_color, font_thickness)
            cv2.putText(side_by_side, "Depth", (depth_label_x, label_y), 
                       font, font_scale, font_color, font_thickness)
            cv2.putText(side_by_side, "Tracking", (tracking_label_x, label_y), 
                       font, font_scale, font_color, font_thickness)
        else:
            # Create a 1x2 grid if no tracking data
            side_by_side = np.hstack((rgb_image, depth_colormap_resized))
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            font_color = (255, 255, 255)
            
            # Calculate text positions
            label_y = 30
            rgb_label_x = w // 2 - 50
            depth_label_x = w + w // 2 - 50
            
            # Add text
            cv2.putText(side_by_side, "RGB", (rgb_label_x, label_y), 
                       font, font_scale, font_color, font_thickness)
            cv2.putText(side_by_side, "Depth", (depth_label_x, label_y), 
                       font, font_scale, font_color, font_thickness)
        
        return side_by_side
    
    def create_tracking_overlay(self, rgb_image, tracking_image):
        """Create a blended overlay of RGB and tracking visualization."""
        # Ensure images are the same size
        if rgb_image.shape != tracking_image.shape:
            tracking_image = cv2.resize(tracking_image, 
                                      (rgb_image.shape[1], rgb_image.shape[0]))
        
        # Create a binary mask where tracking has non-black pixels
        non_black_mask = np.any(tracking_image > 30, axis=2)
        
        # Create output image
        overlay = rgb_image.copy()
        
        # Blend tracking information using the mask
        alpha = 0.7  # Transparency factor
        overlay[non_black_mask] = cv2.addWeighted(
            rgb_image[non_black_mask].reshape(-1, 3), 1-alpha, 
            tracking_image[non_black_mask].reshape(-1, 3), alpha, 0).reshape(-1, 3)
        
        return overlay
    
    def create_point_cloud(self, rgb_image, depth_image):
        """Create a point cloud from RGB and depth data."""
        # Check if we have camera info to do proper 3D projection
        # For now, use approximate values typical for depth cameras
        fx = 525.0  # Focal length x
        fy = 525.0  # Focal length y
        cx = depth_image.shape[1] / 2  # Center x
        cy = depth_image.shape[0] / 2  # Center y
        
        # Convert depth to meters if needed
        if np.median(depth_image[depth_image > 0]) > 1000:
            depth_meters = depth_image / 1000.0
        else:
            depth_meters = depth_image.copy()
        
        # Create point cloud header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_depth_optical_frame"
        
        # Point cloud fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        
        # Downsample for performance
        step = self.point_cloud_downsample
        height = depth_meters.shape[0] // step
        width = depth_meters.shape[1] // step
        
        # Create point cloud data
        points = []
        for v in range(0, depth_meters.shape[0], step):
            for u in range(0, depth_meters.shape[1], step):
                depth = depth_meters[v, u]
                
                # Skip invalid depth
                if depth <= 0 or depth > 10.0:
                    continue
                
                # Calculate 3D point
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                
                # Get color
                if u < rgb_image.shape[1] and v < rgb_image.shape[0]:
                    color = rgb_image[v, u]
                    r, g, b = color[2], color[1], color[0]  # BGR to RGB
                else:
                    r, g, b = 255, 255, 255  # White for out of bounds
                
                # Pack RGB into a single integer
                rgb_packed = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                
                # Add point
                points.append([x, y, z, rgb_packed])
        
        # Convert to numpy array
        point_data = np.array(points, dtype=np.float32).tobytes()
        
        # Create point cloud message
        msg = PointCloud2(
            header=header,
            height=1,
            width=len(points),
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=16,  # 4 fields * 4 bytes
            row_step=16 * len(points),
            data=point_data
        )
        
        return msg

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = Visualization()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in visualization node: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()