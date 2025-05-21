#!/usr/bin/env python3
# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modified for Jetson AGX Xavier compatibility

import cv2
import numpy as np
from typing import List, Tuple
import time
import os

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import KeyPoint3DArray
from yolo_msgs.msg import BoundingBox3D


class Detect3DNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        # Xavier optimization: Set environment variables
        os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS threads for better performance
        
        # Xavier optimization: Set up threading limits before imports
        cv2.setNumThreads(2)  # Limit OpenCV threads for Xavier
        
        # parameters
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter(
            "depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT
        )
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        # Xavier-specific performance parameters
        self.declare_parameter("process_every_n_frame", 1)  # Process every Nth frame
        self.declare_parameter("roi_downsample", 1)  # Downsample factor for depth ROI
        self.declare_parameter("max_points_per_mask", 100)  # Max number of mask points to process
        self.declare_parameter("use_median_depth", True)  # Whether to use median depth (more stable but slower)
        self.declare_parameter("debug_mode", False)  # Enable debug logging
        self.declare_parameter("keypoint_downsample", 2)  # Downsample keypoints for 3D conversion

        # aux
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=2.0))  # Smaller buffer for Xavier
        self.cv_bridge = CvBridge()
        
        # Processing tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.processing_fps = 0
        self.last_fps_update = time.time()
        self.last_transform = None  # Cache last transform for optimization

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.maximum_detection_threshold = (
            self.get_parameter("maximum_detection_threshold")
            .get_parameter_value()
            .double_value
        )
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
        dimg_reliability = (
            self.get_parameter("depth_image_reliability")
            .get_parameter_value()
            .integer_value
        )
        
        # Get Xavier-specific parameters
        self.process_every_n_frame = self.get_parameter("process_every_n_frame").get_parameter_value().integer_value
        self.roi_downsample = self.get_parameter("roi_downsample").get_parameter_value().integer_value
        self.max_points_per_mask = self.get_parameter("max_points_per_mask").get_parameter_value().integer_value
        self.use_median_depth = self.get_parameter("use_median_depth").get_parameter_value().bool_value
        self.debug_mode = self.get_parameter("debug_mode").get_parameter_value().bool_value
        self.keypoint_downsample = self.get_parameter("keypoint_downsample").get_parameter_value().integer_value

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dinfo_reliability = (
            self.get_parameter("depth_info_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)
        
        # Debug publisher for Xavier
        if self.debug_mode:
            self._debug_pub = self.create_publisher(Image, "depth_debug", 1)
        else:
            self._debug_pub = None

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured with Xavier optimizations")
        self.get_logger().info(f"Processing every {self.process_every_n_frame} frame, ROI downsample: {self.roi_downsample}")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
        self.depth_sub = message_filters.Subscriber(
            self, Image, "depth_image", qos_profile=self.depth_image_qos_profile
        )
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info", qos_profile=self.depth_info_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections"
        )

        # Xavier optimization: Use a larger slop value for message synchronization
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 
            queue_size=5,  # Smaller queue for Xavier memory
            slop=0.8  # Larger slop (800ms) for better matching on Xavier
        )
        self._synchronizer.registerCallback(self.on_detections)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.depth_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener
        self.tf_buffer = None

        self.destroy_publisher(self._pub)
        
        if self._debug_pub:
            self.destroy_publisher(self._debug_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")
        
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> None:
        try:
            # Xavier optimization: Skip frames for better performance
            self.frame_count += 1
            if self.frame_count % self.process_every_n_frame != 0:
                return
                
            start_time = time.time()

            new_detections_msg = DetectionArray()
            new_detections_msg.header = detections_msg.header
            
            # Check if there are detections
            if not detections_msg.detections:
                self._pub.publish(new_detections_msg)
                return
            
            # Get transform (with caching optimization for Xavier)
            transform = None
            try:
                # Check if we can use cached transform (within 0.5 seconds)
                if (self.last_transform is not None and 
                    self.last_transform[0] == depth_info_msg.header.frame_id and
                    time.time() - self.last_transform[3] < 0.5):
                    transform = (self.last_transform[1], self.last_transform[2])
                else:
                    transform = self.get_transform(depth_info_msg.header.frame_id)
                    if transform:
                        # Cache the transform with timestamp
                        self.last_transform = (
                            depth_info_msg.header.frame_id,
                            transform[0],
                            transform[1],
                            time.time()
                        )
            except Exception as e:
                self.get_logger().warn(f"Transform error: {str(e)}")
                transform = None

            if transform is None:
                self._pub.publish(new_detections_msg)
                return

            # Convert depth image
            try:
                depth_image = self.cv_bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding="passthrough"
                )
            except Exception as e:
                self.get_logger().error(f"Error converting depth image: {str(e)}")
                self._pub.publish(new_detections_msg)
                return
                
            # Process detections to 3D
            new_detections = []
            for detection in detections_msg.detections:
                try:
                    # Convert to 3D
                    bbox3d = self.convert_bb_to_3d(depth_image, depth_info_msg, detection)

                    if bbox3d is not None:
                        # Create a copy of the detection message for modification
                        det_copy = Detection()
                        det_copy.class_id = detection.class_id
                        det_copy.class_name = detection.class_name
                        det_copy.score = detection.score
                        det_copy.id = detection.id
                        det_copy.bbox = detection.bbox
                        det_copy.detection_time_ms = detection.detection_time_ms
                        
                        # Transform to target frame and set 3D bbox
                        bbox3d = Detect3DNode.transform_3d_box(bbox3d, transform[0], transform[1])
                        bbox3d.frame_id = self.target_frame
                        det_copy.bbox3d = bbox3d

                        # Process keypoints if available and Xavier won't be overloaded
                        if detection.keypoints.data and len(detection.keypoints.data) < 30:
                            try:
                                keypoints3d = self.convert_keypoints_to_3d(
                                    depth_image, depth_info_msg, detection
                                )
                                keypoints3d = Detect3DNode.transform_3d_keypoints(
                                    keypoints3d, transform[0], transform[1]
                                )
                                keypoints3d.frame_id = self.target_frame
                                det_copy.keypoints3d = keypoints3d
                            except Exception as e:
                                self.get_logger().warn(f"Error processing keypoints: {str(e)}")
                        
                        # Add to results
                        new_detections.append(det_copy)
                except Exception as e:
                    self.get_logger().warn(f"Error processing detection to 3D: {str(e)}")
                    continue
                    
            # Set detections and publish
            new_detections_msg.detections = new_detections
            self._pub.publish(new_detections_msg)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Update FPS calculation periodically
            if time.time() - self.last_fps_update > 1.0:
                elapsed = time.time() - self.last_fps_update
                frames_processed = self.frame_count / self.process_every_n_frame
                if elapsed > 0 and frames_processed > 0:
                    self.processing_fps = frames_processed / elapsed
                self.frame_count = 0
                self.last_fps_update = time.time()
                
                # Log performance occasionally
                self.get_logger().info(
                    f"Detect3D: {self.processing_fps:.1f} FPS, " +
                    f"Processing time: {processing_time*1000:.1f}ms, " +
                    f"Objects: {len(new_detections)}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error in on_detections: {str(e)}")
            # Try to publish empty message for robustness
            self._pub.publish(DetectionArray(header=detections_msg.header))

    def convert_bb_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> BoundingBox3D:
        try:
            # Get bounding box info from the 2D detection
            center_x = int(detection.bbox.center.position.x)
            center_y = int(detection.bbox.center.position.y)
            size_x = int(detection.bbox.size.x)
            size_y = int(detection.bbox.size.y)
            
            # Skip invalid boxes
            if size_x <= 0 or size_y <= 0:
                return None
            
            # Xavier optimization: Make sure coordinates are in bounds
            if (center_x < 0 or center_y < 0 or 
                center_x >= depth_image.shape[1] or 
                center_y >= depth_image.shape[0]):
                return None
            
            # Get the depth data from either mask or bounding box
            if detection.mask.data and len(detection.mask.data) > 2:
                # Process using segmentation mask (with Xavier optimizations)
                
                # Sample mask points for performance on Xavier
                mask_points = detection.mask.data
                if len(mask_points) > self.max_points_per_mask:
                    # Downsample mask points for Xavier
                    step = max(1, len(mask_points) // self.max_points_per_mask)
                    mask_points = mask_points[::step]
                
                # Create mask and extract ROI
                mask_array = np.array([[int(ele.x), int(ele.y)] for ele in mask_points])
                
                # Create bounds-checked mask
                mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
                
                # Check if mask array has points and is in bounds before filling
                if len(mask_array) >= 3:
                    try:
                        # Create polygon mask
                        cv2.fillPoly(mask, [np.array(mask_array, dtype=np.int32)], 255)
                        
                        # Apply mask to extract depth values
                        roi = cv2.bitwise_and(depth_image, depth_image, mask=mask)
                        # Get non-zero values
                        roi = roi[roi > 0]
                    except Exception as e:
                        # Fallback to bounding box method if mask fails
                        self.get_logger().warn(f"Mask error, using bbox: {str(e)}")
                        roi = self.get_bbox_roi(depth_image, center_x, center_y, size_x, size_y)
                else:
                    # Fallback to bounding box method if not enough mask points
                    roi = self.get_bbox_roi(depth_image, center_x, center_y, size_x, size_y)
            else:
                # Process using bounding box
                roi = self.get_bbox_roi(depth_image, center_x, center_y, size_x, size_y)

            # Convert to meters
            roi = roi / self.depth_image_units_divisor
            
            # Early return if no valid depth data
            if not np.any(roi):
                return None

            # Get center depth for the 3D position
            if detection.mask.data and len(detection.mask.data) > 2:
                # For masked objects use median of valid points
                if self.use_median_depth:
                    bb_center_z_coord = np.median(roi[roi > 0])
                else:
                    # Use mean for faster but potentially less robust results
                    bb_center_z_coord = np.mean(roi[roi > 0])
            else:
                # For simple box, just use the center point depth (faster)
                center_depth = depth_image[int(center_y)][int(center_x)] / self.depth_image_units_divisor
                if center_depth > 0:
                    bb_center_z_coord = center_depth
                else:
                    # If center point has no depth, fall back to median of ROI
                    valid_depths = roi[roi > 0]
                    if len(valid_depths) > 0:
                        bb_center_z_coord = np.median(valid_depths)
                    else:
                        return None

            # Filter depth values too far from the center depth for better object boundaries
            z_diff = np.abs(roi - bb_center_z_coord)
            mask_z = z_diff <= self.maximum_detection_threshold
            
            # Skip if no depth values within threshold
            if not np.any(mask_z):
                return None

            # Calculate Z bounds for object depth
            filtered_roi = roi[mask_z]
            z_min, z_max = np.min(filtered_roi), np.max(filtered_roi)
            z = (z_max + z_min) / 2  # Use middle of depth range for stability

            # Return early if invalid depth
            if z <= 0:
                return None

            # Project from image to world space
            k = depth_info.k
            px, py, fx, fy = k[2], k[5], k[0], k[4]
            x = z * (center_x - px) / fx
            y = z * (center_y - py) / fy
            
            # Calculate 3D size
            w = z * (size_x / fx)
            h = z * (size_y / fy)

            # Create 3D bounding box message
            msg = BoundingBox3D()
            msg.center.position.x = x
            msg.center.position.y = y
            msg.center.position.z = z
            msg.size.x = w
            msg.size.y = h
            msg.size.z = float(z_max - z_min)

            return msg
            
        except Exception as e:
            self.get_logger().warn(f"Error in convert_bb_to_3d: {str(e)}")
            return None
            
    def get_bbox_roi(self, depth_image, center_x, center_y, size_x, size_y):
        """Extract ROI from depth image using bounding box with Xavier optimizations"""
        # Calculate box bounds with bounds checking
        u_min = max(0, center_x - size_x // 2)
        u_max = min(depth_image.shape[1] - 1, center_x + size_x // 2)
        v_min = max(0, center_y - size_y // 2)
        v_max = min(depth_image.shape[0] - 1, center_y + size_y // 2)
        
        # Skip invalid boxes
        if u_min >= u_max or v_min >= v_max:
            return np.array([])
            
        # Xavier optimization: Downsample ROI for performance
        if self.roi_downsample > 1:
            # Extract ROI with downsampling
            roi = depth_image[v_min:v_max:self.roi_downsample, u_min:u_max:self.roi_downsample]
        else:
            # Extract full ROI
            roi = depth_image[v_min:v_max, u_min:u_max]
            
        return roi.flatten()  # Flatten for efficient processing

    def convert_keypoints_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> KeyPoint3DArray:
        try:
            # Xavier optimization: Downsample keypoints for performance
            keypoints_data = detection.keypoints.data
            if self.keypoint_downsample > 1:
                keypoints_data = keypoints_data[::self.keypoint_downsample]
            
            # Skip if no keypoints after downsampling
            if not keypoints_data:
                return KeyPoint3DArray()

            # build an array of 2d keypoints
            keypoints_2d = np.array(
                [[p.point.x, p.point.y] for p in keypoints_data], dtype=np.int16
            )
            
            # Xavier optimization: Bounds checking
            u = np.array(keypoints_2d[:, 1])
            v = np.array(keypoints_2d[:, 0])
            
            # Filter keypoints that are out of bounds
            valid_mask = (u >= 0) & (u < depth_image.shape[0]) & (v >= 0) & (v < depth_image.shape[1])
            if not np.any(valid_mask):
                return KeyPoint3DArray()
                
            # Only process valid keypoints
            u = u[valid_mask]
            v = v[valid_mask]
            valid_kps = np.array(keypoints_data)[valid_mask]

            # sample depth image and project to 3D
            z = depth_image[u, v]
            
            # Get camera intrinsics
            k = depth_info.k
            px, py, fx, fy = k[2], k[5], k[0], k[4]
            
            # Project to 3D space
            x = z * (v - px) / fx
            y = z * (u - py) / fy
            
            # Stack into 3D points array and convert to meters
            points_3d = np.dstack([x, y, z]).reshape(-1, 3) / self.depth_image_units_divisor

            # generate message
            msg_array = KeyPoint3DArray()
            
            for i, (p, d) in enumerate(zip(points_3d, valid_kps)):
                # Skip points with invalid depth
                if np.any(np.isnan(p)) or p[2] <= 0:
                    continue
                    
                # Create keypoint message
                msg = KeyPoint3D()
                msg.point.x = float(p[0])
                msg.point.y = float(p[1])
                msg.point.z = float(p[2])
                msg.id = d.id
                msg.score = d.score
                msg_array.data.append(msg)

            return msg_array
            
        except Exception as e:
            self.get_logger().warn(f"Error in convert_keypoints_to_3d: {str(e)}")
            return KeyPoint3DArray()

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from image frame to target_frame
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> BoundingBox3D:
        try:
            # Xavier optimization: Use simplified transform computations
            
            # position
            position = Detect3DNode.qv_mult(
                rotation,
                np.array([
                    bbox.center.position.x,
                    bbox.center.position.y,
                    bbox.center.position.z,
                ])
            ) + translation

            bbox.center.position.x = position[0]
            bbox.center.position.y = position[1]
            bbox.center.position.z = position[2]

            # size - avoid unnecessary quaternion rotation for size (approximate)
            # For more accurate but slower result, use: size = Detect3DNode.qv_mult(rotation, ...)
            bbox.size.x = abs(bbox.size.x)
            bbox.size.y = abs(bbox.size.y)
            bbox.size.z = abs(bbox.size.z)

            return bbox
        except Exception as e:
            # Silent failure with original bbox
            return bbox

    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:
        try:
            # Xavier optimization: Batch transform keypoints for better performance
            if not keypoints.data:
                return keypoints
                
            # Extract point data
            points = np.array([
                [point.point.x, point.point.y, point.point.z]
                for point in keypoints.data
            ])
            
            # Transform points in batch
            transformed_points = np.zeros_like(points)
            for i in range(len(points)):
                transformed_points[i] = Detect3DNode.qv_mult(rotation, points[i]) + translation
            
            # Update keypoints with transformed positions
            for i, point in enumerate(keypoints.data):
                point.point.x = transformed_points[i, 0]
                point.point.y = transformed_points[i, 1]
                point.point.z = transformed_points[i, 2]

            return keypoints
        except Exception as e:
            # Silent failure with original keypoints
            return keypoints

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Optimized quaternion-vector multiplication for Xavier"""
        try:
            # Type conversion with better error handling
            q = np.array(q, dtype=np.float32)  # Use float32 for Xavier
            v = np.array(v, dtype=np.float32)
            
            # Extract quaternion components
            q_w = q[0]
            q_vec = q[1:]
            
            # Optimized quaternion vector multiplication
            uv = np.cross(q_vec, v)
            uuv = np.cross(q_vec, uv)
            
            # Final calculation
            return v + 2.0 * (uv * q_w + uuv)
        except Exception as e:
            # Fallback for error cases
            return v  # Return original vector if transformation fails


def main():
    rclpy.init()
    
    try:
        node = Detect3DNode()
        node.trigger_configure()
        node.trigger_activate()
        
        # Use single-threaded executor for Xavier stability
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in detect_3d_node: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()