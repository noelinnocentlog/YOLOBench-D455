#!/usr/bin/env python3
# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modified for Jetson AGX Xavier compatibility

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import cv2
import numpy as np
import message_filters
from cv_bridge import CvBridge
import time
import os
import gc

import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class TrackingNode(Node):
    def __init__(self):
        super().__init__("tracking_node")

        # Xavier optimization: Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0
        os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS threads
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"  # Lower memory fragmentation
        
        # Limit OpenCV threads for Xavier
        cv2.setNumThreads(2)
        
        # Parameters
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        # Xavier-specific parameters
        self.declare_parameter("process_every_n_frame", 1)  # Process every Nth frame
        self.declare_parameter("tracker_fps", 30.0)  # Tracker FPS setting
        self.declare_parameter("downscale_factor", 1.0)  # Downscale images for processing
        self.declare_parameter("performance_monitor", True)  # Monitor and log performance
        self.declare_parameter("max_objects", 20)  # Maximum objects to track for performance
        self.declare_parameter("use_cuda", True)  # Whether to use CUDA for tracking
        
        self.cv_bridge = CvBridge()
        
        # Performance monitoring
        self.frame_count = 0
        self.processed_count = 0
        self.total_processing_time = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Initialize
        self.init_parameters()
        self.init_communications()
        self.init_tracker()
        
        self.get_logger().info("TrackingNode initialized successfully")

    def init_parameters(self):
        # Get parameters
        self.tracker_name = self.get_parameter("tracker").get_parameter_value().string_value
        self.image_reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        
        # Xavier-specific parameters
        self.process_every_n_frame = self.get_parameter("process_every_n_frame").get_parameter_value().integer_value
        self.tracker_fps = self.get_parameter("tracker_fps").get_parameter_value().double_value
        self.downscale_factor = self.get_parameter("downscale_factor").get_parameter_value().double_value
        self.performance_monitor = self.get_parameter("performance_monitor").get_parameter_value().bool_value
        self.max_objects = self.get_parameter("max_objects").get_parameter_value().integer_value
        self.use_cuda = self.get_parameter("use_cuda").get_parameter_value().bool_value

    def init_communications(self):
        # Setup QoS profile
        image_qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Initialize publisher
        self._pub = self.create_publisher(DetectionArray, "tracking", 10)
        
        # Performance overlay publisher if enabled
        if self.performance_monitor:
            self._perf_pub = self.create_publisher(Image, "tracking_performance", 1)

        # Create subscribers
        self.image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10
        )

        # Xavier optimization: Use a larger slop value for better synchronization
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 
            queue_size=5,  # Smaller queue for Xavier memory
            slop=0.8  # Larger slop (800ms) for better message matching on Xavier
        )
        self._synchronizer.registerCallback(self.detections_cb)

    def init_tracker(self):
        try:
            # Before tracker creation, optimize Xavier memory
            if torch.cuda.is_available() and self.use_cuda:
                torch.cuda.empty_cache()
                
                # Log GPU info
                self.get_logger().info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                self.get_logger().info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                
                # Set device
                self.device = "cuda:0"
            else:
                self.device = "cpu"
                self.get_logger().warn("CUDA not available, using CPU for tracking")
                
            # Create tracker with Xavier optimizations
            self.tracker = self.create_tracker(
                self.tracker_name, 
                frame_rate=self.tracker_fps
            )
            
            # Warm-up the tracker for better initial performance
            self.warm_up_tracker()
            
        except Exception as e:
            self.get_logger().error(f"Error initializing tracker: {str(e)}")
            # Create simple fallback tracker
            self.tracker = BYTETracker(
                IterableSimpleNamespace(track_thresh=0.25, track_buffer=30, match_thresh=0.8), 
                frame_rate=30
            )

    def create_tracker(self, tracker_yaml: str, frame_rate: float = 30.0) -> BaseTrack:
        """Create tracker with Xavier-specific optimizations"""
        try:
            TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
            check_requirements("lap")  # for linear_assignment

            # Check if tracker file exists
            tracker = check_yaml(tracker_yaml)
            cfg = IterableSimpleNamespace(**yaml_load(tracker))

            # Xavier: Use only supported trackers
            if cfg.tracker_type not in ["bytetrack", "botsort"]:
                self.get_logger().warn(f"Unsupported tracker type: {cfg.tracker_type}, falling back to bytetrack")
                cfg.tracker_type = "bytetrack"
                
            # Xavier optimization: For BoTSORT, use lighter settings
            if cfg.tracker_type == "botsort":
                # Use simpler feature extractor for Xavier
                cfg.botsort_feature_extractors = "mobilenet_x1_0"
                # Disable appearance matching for speed
                cfg.botsort_match_thresh = 0.8
                
            # Create tracker with specified frame rate for better adaptation to actual FPS
            tracker = TRACKER_MAP[cfg.tracker_type](
                args=cfg, 
                frame_rate=frame_rate
            )
            
            self.get_logger().info(f"Created {cfg.tracker_type} tracker with frame_rate={frame_rate}")
            return tracker
            
        except Exception as e:
            self.get_logger().error(f"Error creating tracker: {str(e)}")
            # Fallback to simple ByteTrack with minimal settings
            self.get_logger().warn(f"Falling back to default ByteTrack")
            return BYTETracker(IterableSimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8), frame_rate=30)

    def warm_up_tracker(self):
        """Warm up the tracker with a dummy frame for better initial performance"""
        if not self.use_cuda:
            return  # Skip warmup if not using CUDA
            
        try:
            # Create dummy image and detection
            dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
            dummy_det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
            
            # Create dummy Boxes object
            dummy_boxes = Boxes(dummy_det, (416, 416))
            
            # Run tracker once with dummy data
            self.tracker.update(dummy_boxes, dummy_img)
            
            self.get_logger().info("Tracker warmed up successfully")
        except Exception as e:
            self.get_logger().warn(f"Tracker warmup failed: {str(e)}")

    def create_performance_overlay(self, cv_image, processing_time):
        """Create a performance statistics overlay for monitoring"""
        if not self.performance_monitor:
            return None
            
        try:
            # Create a copy for overlay
            overlay = cv_image.copy()
            
            # Calculate metrics
            current_time = time.time()
            if current_time - self.last_fps_time > 1.0:
                self.fps = self.processed_count / (current_time - self.last_fps_time)
                self.processed_count = 0
                self.last_fps_time = current_time
                
            avg_time = self.total_processing_time / max(1, self.processed_count) * 1000
            
            # Draw background for text
            cv2.rectangle(overlay, (10, 10), (280, 90), (0, 0, 0), -1)
            
            # Draw stats
            cv2.putText(overlay, f"Tracking FPS: {self.fps:.1f}", 
                      (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay, f"Process Time: {processing_time*1000:.1f}ms", 
                      (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay, f"Frames Processed: {self.frame_count}", 
                      (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return overlay
        except Exception as e:
            self.get_logger().warn(f"Error creating performance overlay: {str(e)}")
            return None

    def detections_cb(self, img_msg: Image, detections_msg: DetectionArray) -> None:
        # Update frame counter for performance monitoring
        self.frame_count += 1
        
        # Xavier optimization: Skip frames if needed
        if self.process_every_n_frame > 1 and (self.frame_count % self.process_every_n_frame != 0):
            # Just republish the original detections to maintain stream
            self._pub.publish(detections_msg)
            return
            
        # Start timing
        start_time = time.time()
        self.processed_count += 1
        
        # Create output message
        tracked_detections_msg = DetectionArray()
        tracked_detections_msg.header = img_msg.header
        
        try:
            # Convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            
            # Xavier optimization: Downscale image for processing if needed
            if self.downscale_factor < 1.0:
                h, w = cv_image.shape[:2]
                new_h, new_w = int(h * self.downscale_factor), int(w * self.downscale_factor)
                process_img = cv2.resize(cv_image, (new_w, new_h))
                # Need to scale detection coordinates too
                scale_factor = self.downscale_factor
            else:
                process_img = cv_image
                scale_factor = 1.0
                
            # Convert to RGB for tracker
            process_img = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)

            # Early return if no detections
            if not detections_msg.detections:
                self._pub.publish(tracked_detections_msg)
                
                # Create performance overlay if enabled
                if self.performance_monitor:
                    processing_time = time.time() - start_time
                    self.total_processing_time += processing_time
                    
                    perf_overlay = self.create_performance_overlay(cv_image, processing_time)
                    if perf_overlay is not None:
                        perf_msg = self.cv_bridge.cv2_to_imgmsg(perf_overlay, encoding="bgr8")
                        perf_msg.header = img_msg.header
                        self._perf_pub.publish(perf_msg)
                return

            # Xavier optimization: Limit number of objects to track
            detections_to_process = detections_msg.detections
            if len(detections_to_process) > self.max_objects:
                # Sort by score and take top N
                detections_to_process = sorted(
                    detections_to_process, 
                    key=lambda x: x.score, 
                    reverse=True
                )[:self.max_objects]
                self.get_logger().debug(f"Limiting to {self.max_objects} highest-scoring objects")

            # Parse detections for tracking
            detection_list = []
            for detection in detections_to_process:
                # Scale coordinates if image was resized
                x_center = detection.bbox.center.position.x * scale_factor
                y_center = detection.bbox.center.position.y * scale_factor
                width = detection.bbox.size.x * scale_factor
                height = detection.bbox.size.y * scale_factor
                
                # Convert to xyxy format for tracker
                detection_list.append([
                    x_center - width / 2,    # x1
                    y_center - height / 2,   # y1
                    x_center + width / 2,    # x2
                    y_center + height / 2,   # y2
                    detection.score,         # confidence
                    detection.class_id,      # class ID
                ])

            # Run tracking if we have detections
            if detection_list:
                # Convert to Boxes format
                det = Boxes(
                    np.array(detection_list), 
                    (process_img.shape[0], process_img.shape[1])
                )
                
                # Run tracker
                tracks = self.tracker.update(det, process_img)

                # Process tracker results
                if len(tracks) > 0:
                    for t in tracks:
                        try:
                            # Get box in tracker format
                            tracked_box = Boxes(t[:-1], (process_img.shape[0], process_img.shape[1]))
                            
                            # Get the original detection this track corresponds to
                            detection_idx = int(t[-1])
                            if detection_idx >= len(detections_to_process):
                                self.get_logger().warn(f"Invalid detection index: {detection_idx}")
                                continue
                                
                            original_detection = detections_to_process[detection_idx]
                            
                            # Create a copy of the detection
                            tracked_detection = Detection()
                            tracked_detection.class_id = original_detection.class_id
                            tracked_detection.class_name = original_detection.class_name
                            tracked_detection.score = original_detection.score
                            tracked_detection.mask = original_detection.mask
                            tracked_detection.keypoints = original_detection.keypoints
                            tracked_detection.bbox3d = original_detection.bbox3d
                            tracked_detection.keypoints3d = original_detection.keypoints3d
                            tracked_detection.detection_time_ms = original_detection.detection_time_ms
                            
                            # Copy the bbox for modification
                            tracked_detection.bbox = original_detection.bbox
                            
                            # Update bounding box with tracked position
                            box = tracked_box.xywh[0]
                            
                            # Scale back if image was resized
                            if scale_factor != 1.0:
                                scaled_x = float(box[0]) / scale_factor
                                scaled_y = float(box[1]) / scale_factor
                                scaled_w = float(box[2]) / scale_factor
                                scaled_h = float(box[3]) / scale_factor
                                
                                tracked_detection.bbox.center.position.x = scaled_x
                                tracked_detection.bbox.center.position.y = scaled_y
                                tracked_detection.bbox.size.x = scaled_w
                                tracked_detection.bbox.size.y = scaled_h
                            else:
                                tracked_detection.bbox.center.position.x = float(box[0])
                                tracked_detection.bbox.center.position.y = float(box[1])
                                tracked_detection.bbox.size.x = float(box[2])
                                tracked_detection.bbox.size.y = float(box[3])

                            # Set track ID
                            if tracked_box.is_track:
                                tracked_detection.id = str(int(tracked_box.id))
                            else:
                                tracked_detection.id = ""

                            # Add to output message
                            tracked_detections_msg.detections.append(tracked_detection)
                        except Exception as e:
                            self.get_logger().warn(f"Error processing track: {str(e)}")
                            continue

            # Publish tracked detections
            self._pub.publish(tracked_detections_msg)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Create and publish performance overlay if enabled
            if self.performance_monitor:
                perf_overlay = self.create_performance_overlay(cv_image, processing_time)
                if perf_overlay is not None:
                    perf_msg = self.cv_bridge.cv2_to_imgmsg(perf_overlay, encoding="bgr8")
                    perf_msg.header = img_msg.header
                    self._perf_pub.publish(perf_msg)
            
            # Log performance occasionally
            if self.performance_monitor and self.frame_count % 100 == 0:
                self.get_logger().info(
                    f"Tracking: {self.fps:.1f} FPS, " +
                    f"Processing: {processing_time*1000:.1f}ms, " +
                    f"Objects: {len(tracked_detections_msg.detections)}"
                )
                
            # Xavier optimization: Clean CUDA memory occasionally
            if self.use_cuda and self.frame_count % 300 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.get_logger().error(f"Error in detection callback: {str(e)}")
            # Try to publish original detections as fallback
            self._pub.publish(detections_msg)


def main():
    rclpy.init()
    
    try:
        # Create node
        node = TrackingNode()
        
        # Start processing
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in tracking node: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()