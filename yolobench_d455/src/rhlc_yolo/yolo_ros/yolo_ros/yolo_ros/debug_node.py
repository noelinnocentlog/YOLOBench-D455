#!/usr/bin/env python3
# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modified for Jetson AGX Xavier compatibility

import cv2
import random
import numpy as np
from typing import Tuple
import time
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class DebugNode(Node):
    def __init__(self):
        super().__init__("debug_node")

        # Set environment variables for Xavier optimization
        os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Prevent OpenBLAS from using too many threads
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU 0 is used
        
        # Limit OpenCV threads
        cv2.setNumThreads(2)

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # params
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        # Performance parameters for Xavier
        self.declare_parameter("enable_markers", True)
        self.declare_parameter("enable_keypoints", True)
        self.declare_parameter("enable_masks", True)
        self.declare_parameter("text_scale", 0.8)
        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("downsample_output", False)
        self.declare_parameter("performance_overlay", True)
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        # Initialize
        self.init_parameters()
        self.init_communications()
        
        self.get_logger().info("DebugNode initialized successfully")

    def init_parameters(self):
        # Get QoS reliability parameter
        reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        
        self.image_qos_profile = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Get xavier-specific parameters
        self.enable_markers = self.get_parameter("enable_markers").get_parameter_value().bool_value
        self.enable_keypoints = self.get_parameter("enable_keypoints").get_parameter_value().bool_value
        self.enable_masks = self.get_parameter("enable_masks").get_parameter_value().bool_value
        self.text_scale = self.get_parameter("text_scale").get_parameter_value().double_value
        self.line_thickness = self.get_parameter("line_thickness").get_parameter_value().integer_value
        self.downsample_output = self.get_parameter("downsample_output").get_parameter_value().bool_value
        self.performance_overlay = self.get_parameter("performance_overlay").get_parameter_value().bool_value

    def init_communications(self):
        # pubs - only create if enabled for Xavier performance
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        
        if self.enable_markers:
            self._bb_markers_pub = self.create_publisher(MarkerArray, "dgb_bb_markers", 5)
            self._kp_markers_pub = self.create_publisher(MarkerArray, "dgb_kp_markers", 5)
        else:
            self._bb_markers_pub = None
            self._kp_markers_pub = None

        # subs
        self.image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=self.image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10
        )

        # Use a larger slop value (0.8 sec) for better message matching on Xavier
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 5, 0.8
        )
        self._synchronizer.registerCallback(self.detections_cb)

    def draw_box(
        self,
        cv_image: np.ndarray,
        detection: Detection,
        color: Tuple[int],
    ) -> np.ndarray:
        try:
            # get detection info
            class_name = detection.class_name
            score = detection.score
            box_msg: BoundingBox2D = detection.bbox
            track_id = detection.id

            # Calculate bounding box coordinates
            min_pt = (
                round(box_msg.center.position.x - box_msg.size.x / 2.0),
                round(box_msg.center.position.y - box_msg.size.y / 2.0),
            )
            max_pt = (
                round(box_msg.center.position.x + box_msg.size.x / 2.0),
                round(box_msg.center.position.y + box_msg.size.y / 2.0),
            )
            
            # Optimization for Xavier: Simplified drawing when theta is close to 0
            if abs(box_msg.center.theta) < 0.01:  # Less than ~0.5 degrees
                # Draw simple rectangle for better performance
                cv2.rectangle(cv_image, min_pt, max_pt, color, self.line_thickness)
            else:
                # For rotated boxes, use the full computation
                # define the four corners of the rectangle
                rect_pts = np.array(
                    [
                        [min_pt[0], min_pt[1]],
                        [max_pt[0], min_pt[1]],
                        [max_pt[0], max_pt[1]],
                        [min_pt[0], max_pt[1]],
                    ]
                )

                # calculate the rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(
                    (box_msg.center.position.x, box_msg.center.position.y),
                    -np.rad2deg(box_msg.center.theta),
                    1.0,
                )

                # rotate the corners of the rectangle
                rect_pts = np.int0(cv2.transform(np.array([rect_pts]), rotation_matrix)[0])

                # Draw the rotated rectangle
                for i in range(4):
                    pt1 = tuple(rect_pts[i])
                    pt2 = tuple(rect_pts[(i + 1) % 4])
                    cv2.line(cv_image, pt1, pt2, color, self.line_thickness)

            # write text (optimized for Xavier)
            # Shorter label to improve drawing speed
            label = f"{class_name}"
            if track_id:
                label += f" ({track_id})"
            label += f" {score:.2f}"  # Show only 2 decimal places for speed
            
            pos = (min_pt[0] + 5, min_pt[1] + 25)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Black background for better readability
            text_size = cv2.getTextSize(label, font, self.text_scale, self.line_thickness)[0]
            cv2.rectangle(cv_image, 
                         (pos[0]-2, pos[1]-text_size[1]-2), 
                         (pos[0]+text_size[0]+2, pos[1]+2), 
                         (0,0,0), -1)
            
            # Draw text
            cv2.putText(cv_image, label, pos, font, self.text_scale, color, self.line_thickness, cv2.LINE_AA)

            return cv_image
        except Exception as e:
            self.get_logger().warn(f"Error drawing box: {str(e)}")
            return cv_image

    def draw_mask(
        self,
        cv_image: np.ndarray,
        detection: Detection,
        color: Tuple[int],
    ) -> np.ndarray:
        try:
            if not self.enable_masks:
                return cv_image
                
            mask_msg = detection.mask
            
            # Skip if no mask data
            if not mask_msg.data:
                return cv_image
                
            # Convert mask points to array for drawing
            mask_array = np.array([[int(ele.x), int(ele.y)] for ele in mask_msg.data])
            
            # Xavier optimization: if too many points, downsample for performance
            if len(mask_array) > 100:
                # Downsample points by taking every Nth point
                mask_array = mask_array[::len(mask_array)//100]
                
            # Skip if not enough points after downsampling
            if len(mask_array) < 3:
                return cv_image
            
            # Create a separate layer for the mask
            try:
                # Xavier optimization: Only create layer if needed
                layer = cv_image.copy()
                # Use a more transparent overlay for better visibility
                alpha = 0.4  # Lower alpha value for transparency
                
                # Fill polygon with color
                cv2.fillPoly(layer, pts=[mask_array], color=color)
                
                # Blend with main image
                cv2.addWeighted(cv_image, 1-alpha, layer, alpha, 0, cv_image)
                
                # Draw outline
                cv2.polylines(
                    cv_image,
                    [mask_array],
                    isClosed=True,
                    color=color,
                    thickness=1,  # Thinner line for better performance
                    lineType=cv2.LINE_AA,
                )
            except Exception as e:
                self.get_logger().warn(f"Error filling mask: {str(e)}")
                # Fall back to just drawing the outline
                cv2.polylines(
                    cv_image,
                    [mask_array],
                    isClosed=True,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                
            return cv_image
        except Exception as e:
            self.get_logger().warn(f"Error drawing mask: {str(e)}")
            return cv_image

    def draw_keypoints(self, cv_image: np.ndarray, detection: Detection) -> np.ndarray:
        try:
            if not self.enable_keypoints:
                return cv_image
                
            keypoints_msg = detection.keypoints
            
            # Check if there are any keypoints
            if not keypoints_msg.data:
                return cv_image

            ann = Annotator(cv_image)

            # Draw keypoints
            kp: KeyPoint2D
            for kp in keypoints_msg.data:
                # Choose color based on keypoint ID
                color_k = (
                    [int(x) for x in ann.kpt_color[kp.id - 1]]
                    if len(keypoints_msg.data) == 17 and kp.id <= 17
                    else colors(kp.id - 1)
                )

                # Draw circle at keypoint location
                cv2.circle(
                    cv_image,
                    (int(kp.point.x), int(kp.point.y)),
                    4,  # Smaller radius for performance
                    color_k,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                
                # For Xavier performance, don't draw keypoint IDs
                # They add significant overhead with little value for most applications

            # Xavier optimization: Only draw main skeleton connections
            # Create helper function for finding keypoints
            def get_pk_pose(kp_id: int) -> Tuple[int]:
                for kp in keypoints_msg.data:
                    if kp.id == kp_id:
                        return (int(kp.point.x), int(kp.point.y))
                return None
                
            # Draw lines between keypoints (skeleton)
            # Only process the first 8 skeleton connections to save computation
            max_skeleton = min(8, len(ann.skeleton))
            for i in range(max_skeleton):
                sk = ann.skeleton[i]
                kp1_pos = get_pk_pose(sk[0])
                kp2_pos = get_pk_pose(sk[1])

                if kp1_pos is not None and kp2_pos is not None:
                    cv2.line(
                        cv_image,
                        kp1_pos,
                        kp2_pos,
                        [int(x) for x in ann.limb_color[i]],
                        thickness=1,  # Thinner lines for performance
                        lineType=cv2.LINE_AA,
                    )

            return cv_image
        except Exception as e:
            self.get_logger().warn(f"Error drawing keypoints: {str(e)}")
            return cv_image

    def create_bb_marker(self, detection: Detection, color: Tuple[int]) -> Marker:
        try:
            bbox3d = detection.bbox3d

            marker = Marker()
            marker.header.frame_id = bbox3d.frame_id

            marker.ns = "yolo_3d"
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.frame_locked = False

            marker.pose.position.x = bbox3d.center.position.x
            marker.pose.position.y = bbox3d.center.position.y
            marker.pose.position.z = bbox3d.center.position.z

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = bbox3d.size.x
            marker.scale.y = bbox3d.size.y
            marker.scale.z = bbox3d.size.z

            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.4

            # Set lifetime to 0.2 seconds
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000
            marker.text = detection.class_name

            return marker
        except Exception as e:
            self.get_logger().warn(f"Error creating bounding box marker: {str(e)}")
            return None

    def create_kp_marker(self, keypoint: KeyPoint3D) -> Marker:
        try:
            marker = Marker()

            marker.ns = "yolo_3d"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.frame_locked = False

            marker.pose.position.x = keypoint.point.x
            marker.pose.position.y = keypoint.point.y
            marker.pose.position.z = keypoint.point.z

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Xavier optimization: Smaller markers for better performance
            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03

            marker.color.r = (1.0 - keypoint.score) * 255.0
            marker.color.g = 0.0
            marker.color.b = keypoint.score * 255.0
            marker.color.a = 0.4

            # Set lifetime to 0.2 seconds
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000
            
            # Don't include text (ID) for improved performance
            # marker.text = str(keypoint.id)

            return marker
        except Exception as e:
            self.get_logger().warn(f"Error creating keypoint marker: {str(e)}")
            return None

    def create_performance_overlay(self, cv_image, processing_time):
        """Add performance statistics overlay"""
        if not self.performance_overlay:
            return cv_image
            
        # Calculate and update FPS
        self.frame_count += 1
        self.total_processing_time += processing_time
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update > 1.0:
            elapsed = current_time - self.last_fps_update
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
        
        # Draw dark background for text
        cv2.rectangle(cv_image, (10, 10), (250, 70), (0, 0, 0), -1)
        
        # Draw performance stats
        avg_time = self.total_processing_time / max(1, self.frame_count) * 1000
        
        cv2.putText(cv_image, f"DebugNode FPS: {self.fps:.1f}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(cv_image, f"Process time: {processing_time*1000:.1f}ms", (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        return cv_image

    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray) -> None:
        start_time = time.time()
        
        try:
            # Convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            
            # For Xavier performance, optionally downsample the input image for processing
            if self.downsample_output and cv_image.shape[0] > 720:  # If very high resolution
                h, w = cv_image.shape[:2]
                # Downsample for processing, but keep original for final output
                scale_factor = 720 / h
                process_img = cv2.resize(cv_image, (int(w * scale_factor), 720))
            else:
                process_img = cv_image
            
            # Initialize marker arrays if enabled
            bb_marker_array = MarkerArray() if self.enable_markers and self._bb_markers_pub else None
            kp_marker_array = MarkerArray() if self.enable_markers and self._kp_markers_pub else None

            # Process each detection
            for detection in detection_msg.detections:
                try:
                    # Get or generate color for this class
                    class_name = detection.class_name

                    if class_name not in self._class_to_color:
                        # Generate consistent colors for classes
                        # Xavier optimization: Use class name hash for more consistent colors
                        hash_val = hash(class_name) % 100
                        r = 55 + (hash_val * 7) % 200
                        g = 55 + (hash_val * 11) % 200
                        b = 55 + (hash_val * 17) % 200
                        self._class_to_color[class_name] = (r, g, b)

                    color = self._class_to_color[class_name]

                    # Draw visualization elements
                    process_img = self.draw_box(process_img, detection, color)
                    
                    if self.enable_masks:
                        process_img = self.draw_mask(process_img, detection, color)
                    
                    if self.enable_keypoints:
                        process_img = self.draw_keypoints(process_img, detection)

                    # Create and add markers if enabled
                    if self.enable_markers:
                        if bb_marker_array is not None and detection.bbox3d.frame_id:
                            marker = self.create_bb_marker(detection, color)
                            if marker:
                                marker.header.stamp = img_msg.header.stamp
                                marker.id = len(bb_marker_array.markers)
                                bb_marker_array.markers.append(marker)

                        if kp_marker_array is not None and detection.keypoints3d.frame_id:
                            # Xavier optimization: Only add every 2nd keypoint for performance
                            for i, kp in enumerate(detection.keypoints3d.data):
                                if i % 2 == 0:  # Add only every 2nd keypoint
                                    marker = self.create_kp_marker(kp)
                                    if marker:
                                        marker.header.frame_id = detection.keypoints3d.frame_id
                                        marker.header.stamp = img_msg.header.stamp
                                        marker.id = len(kp_marker_array.markers)
                                        kp_marker_array.markers.append(marker)
                except Exception as e:
                    self.get_logger().warn(f"Error processing detection: {str(e)}")
                    continue

            # Calculate processing time and add performance overlay
            processing_time = time.time() - start_time
            process_img = self.create_performance_overlay(process_img, processing_time)
            
            # Copy processed image back to original if downsampled
            if self.downsample_output and process_img is not cv_image:
                # Upscale the processed image back to original size
                output_img = cv2.resize(process_img, (cv_image.shape[1], cv_image.shape[0]))
            else:
                output_img = process_img
                
            # publish debug image
            self._dbg_pub.publish(
                self.cv_bridge.cv2_to_imgmsg(output_img, encoding="bgr8", header=img_msg.header)
            )
            
            # publish markers if enabled
            if self.enable_markers:
                if bb_marker_array is not None and self._bb_markers_pub:
                    self._bb_markers_pub.publish(bb_marker_array)
                if kp_marker_array is not None and self._kp_markers_pub:
                    self._kp_markers_pub.publish(kp_marker_array)
                    
            # Log performance occasionally
            if self.frame_count % 100 == 0:
                self.get_logger().info(f"Debug node performance: {self.fps:.1f} FPS")
                
        except Exception as e:
            self.get_logger().error(f"Error in detection callback: {str(e)}")


def main():
    rclpy.init()
    
    try:
        # Create node
        node = DebugNode()
        
        # Start processing
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in debug node: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()