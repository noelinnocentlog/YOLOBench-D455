#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import json


@dataclass
class TrackedObject:
    """Class for storing tracked object information across frames"""
    id: int                          # Unique ID for this object
    class_name: str                  # YOLO class name
    last_seen: float                 # Timestamp when last detected
    positions: deque                 # Queue of (timestamp, x, y, w, h, depth) tuples
    velocities: deque                # Queue of velocity measurements (pixels/sec, m/sec)
    risk_scores: deque               # Queue of risk scores
    color: Tuple[int, int, int]      # Color for visualization
    avg_depth: float = 0.0           # Moving average of depth
    depth_variance: float = 0.0      # Variance in depth measurements
    predicted_position: Optional[Tuple[int, int, int, int]] = None  # Predicted next position
    zone: str = "unknown"            # Current zone based on depth
    
    def update_depth_stats(self):
        """Update moving average depth and variance"""
        if len(self.positions) > 0:
            depths = [pos[5] for pos in self.positions if pos[5] > 0]
            if depths:
                self.avg_depth = np.mean(depths)
                if len(depths) > 1:
                    self.depth_variance = np.var(depths)
        
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity in pixels/sec and m/sec"""
        if len(self.velocities) > 0:
            return self.velocities[-1]
        return (0.0, 0.0)
    
    def get_risk_score(self) -> float:
        """Get current risk score"""
        if len(self.risk_scores) > 0:
            return self.risk_scores[-1]
        return 0.0


class DistanceZoneTrackingNode(Node):
    def __init__(self):
        super().__init__('distance_zone_tracking_node')
        self.bridge = CvBridge()
        
        # ---- Parameters ----
        self.declare_parameter('min_depth', 0.1)             # Min valid depth in meters
        self.declare_parameter('max_depth', 10.0)            # Max valid depth in meters
        self.declare_parameter('depth_method', 'median')     # Options: 'median', 'mean', 'min'
        self.declare_parameter('roi_scale', 0.5)             # Scale factor for ROI relative to bounding box
        self.declare_parameter('tracking_history', 20)       # Number of frames to keep for tracking
        self.declare_parameter('max_tracking_age', 1.0)      # Max seconds before dropping track
        self.declare_parameter('iou_threshold', 0.3)         # IoU threshold for matching
        self.declare_parameter('risk_depth_weight', 0.6)     # w1: Weight for depth in risk calculation
        self.declare_parameter('risk_velocity_weight', 0.15) # w2: Weight for velocity in risk calculation
        self.declare_parameter('risk_class_weight', 0.15)    # w3: Weight for class priority in risk calculation
        self.declare_parameter('risk_size_weight', 0.1)      # w4: Weight for object size in risk calculation
        self.declare_parameter('class_priority_file', '')    # Path to JSON file with class priorities
        
        # Distance zone thresholds (in meters)
        self.declare_parameter('red_zone_threshold', 1.5)     # Objects closer than this are in red zone
        self.declare_parameter('yellow_zone_threshold', 3.0)  # Objects closer than this are in yellow zone
                                                              # Objects beyond this are in green zone
        
        # Get parameters
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.depth_method = self.get_parameter('depth_method').value
        self.roi_scale = self.get_parameter('roi_scale').value
        self.tracking_history = self.get_parameter('tracking_history').value
        self.max_tracking_age = self.get_parameter('max_tracking_age').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.risk_depth_weight = self.get_parameter('risk_depth_weight').value
        self.risk_velocity_weight = self.get_parameter('risk_velocity_weight').value 
        self.risk_class_weight = self.get_parameter('risk_class_weight').value
        self.risk_size_weight = self.get_parameter('risk_size_weight').value
        self.class_priority_file = self.get_parameter('class_priority_file').value
        
        # Get zone thresholds
        self.red_zone_threshold = self.get_parameter('red_zone_threshold').value
        self.yellow_zone_threshold = self.get_parameter('yellow_zone_threshold').value
        
        # Load class priorities
        self.class_priorities = self.load_class_priorities()
        
        # Define distance zone colors (BGR format for OpenCV)
        self.zone_colors = {
            "red": (0, 0, 255),        # Red for close objects (high risk)
            "yellow": (0, 255, 255),   # Yellow for medium distance
            "green": (0, 255, 0),      # Green for far objects (low risk)
            "unknown": (180, 180, 180) # Gray for unknown depth
        }
        
        # Tracking variables
        self.tracked_objects = {}    # Dictionary of tracked objects by ID
        self.next_id = 0             # Next object ID to assign
        self.last_frame_time = None  # Timestamp of last processed frame
        self.frame_counter = 0       # Counter for processed frames
        
        # Create subscribers using message_filters
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
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
        
        # Publishers
        self.image_pub = self.create_publisher(Image, '/yolo/depth_tracking_image', 10)
        
        # Print initialization status
        self.get_logger().info("DistanceZoneTrackingNode initialized with parameters:")
        self.get_logger().info(f"- min_depth: {self.min_depth}m, max_depth: {self.max_depth}m")
        self.get_logger().info(f"- depth_method: {self.depth_method}, roi_scale: {self.roi_scale}")
        self.get_logger().info(f"- tracking_history: {self.tracking_history}, max_tracking_age: {self.max_tracking_age}s")
        self.get_logger().info(f"- zone thresholds - red: {self.red_zone_threshold}m, yellow: {self.yellow_zone_threshold}m")
        self.get_logger().info("Waiting for data...")

    def load_class_priorities(self) -> Dict[str, float]:
        """Load class priorities from file or return default values"""
        default_priorities = {
            'person': 1.0, 
            'bicycle': 0.8, 
            'car': 0.9, 
            'motorcycle': 0.85, 
            'bus': 0.95, 
            'truck': 0.9,
            'animal': 0.85,
            'chair': 0.4, 
            'sofa': 0.3,
            'bed': 0.3,
            'dining table': 0.3,
            'toilet': 0.2,
            'tv': 0.2,
            'laptop': 0.2,
            'keyboard': 0.1,
            'cell phone': 0.2,
            'microwave': 0.2,
            'oven': 0.2,
            'toaster': 0.2,
            'refrigerator': 0.3,
            'book': 0.1,
            'clock': 0.1
        }
        
        # Try to load from file if provided
        if self.class_priority_file:
            try:
                with open(self.class_priority_file, 'r') as f:
                    loaded_priorities = json.load(f)
                self.get_logger().info(f"Loaded class priorities from {self.class_priority_file}")
                return loaded_priorities
            except Exception as e:
                self.get_logger().warn(f"Failed to load class priorities from file: {str(e)}")
                self.get_logger().warn(f"Using default class priorities")
        
        return default_priorities

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
        # Handle empty input
        if len(depth_values) == 0:
            return np.array([])
            
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
        
        # Calculate ROI coordinates with bounds checking
        roi_x1 = max(0, center_x - roi_w // 2)
        roi_y1 = max(0, center_y - roi_h // 2)
        roi_x2 = min(img_w - 1, roi_x1 + roi_w)
        roi_y2 = min(img_h - 1, roi_y1 + roi_h)
        
        # Check for invalid ROI
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return 0.0, 0.0
            
        # Extract ROI from depth image
        try:
            depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
        except IndexError:
            self.get_logger().warn(f"Invalid ROI: ({roi_x1}:{roi_x2}, {roi_y1}:{roi_y2}) for image shape {depth_image.shape}")
            return 0.0, 0.0
        
        # Filter depth values
        valid_depths = self.filter_depth_values(depth_roi.flatten())
        
        if len(valid_depths) == 0:
            return 0.0, 0.0  # No valid depth, zero confidence
        
        # Calculate confidence as ratio of valid pixels
        roi_area = max(1, (roi_x2 - roi_x1) * (roi_y2 - roi_y1))
        confidence = len(valid_depths) / roi_area
        
        # Calculate depth based on selected method
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

    def determine_zone(self, depth):
        """
        Determine which zone an object belongs to based on its depth.
        
        Args:
            depth: Object depth in meters
            
        Returns:
            Zone name: "red", "yellow", "green", or "unknown"
        """
        if depth <= 0:
            return "unknown"
        elif depth < self.red_zone_threshold:
            return "red"
        elif depth < self.yellow_zone_threshold:
            return "yellow"
        else:
            return "green"

    def calculate_iou(self, box1, box2):
        """
        Calculate intersection over union between two bounding boxes.
        
        Args:
            box1: (x1, y1, w1, h1)
            box2: (x2, y2, w2, h2)
            
        Returns:
            IoU value
        """
        # Convert to (x1, y1, x2, y2) format
        box1_x1, box1_y1, box1_w, box1_h = box1
        box1_x2, box1_y2 = box1_x1 + box1_w, box1_y1 + box1_h
        
        box2_x1, box2_y1, box2_w, box2_h = box2
        box2_x2, box2_y2 = box2_x1 + box2_w, box2_y1 + box2_h
        
        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = box1_w * box1_h
        box2_area = box2_w * box2_h
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / max(union_area, 1e-6)
        
        return iou

    def calculate_velocity(self, obj, current_time, x, y, depth):
        """
        Calculate velocity based on position history.
        
        Args:
            obj: TrackedObject
            current_time: Current timestamp
            x, y: Current center position
            depth: Current depth
            
        Returns:
            (velocity_pixels_per_sec, velocity_meters_per_sec)
        """
        if len(obj.positions) < 2:
            return (0.0, 0.0)
            
        # Get previous position with valid depth
        prev_positions = [(ts, cx, cy, d) for ts, cx, cy, _, _, d in obj.positions if d > 0]
        if not prev_positions:
            return (0.0, 0.0)
            
        prev_time, prev_x, prev_y, prev_depth = prev_positions[-1]
        
        # Calculate time difference
        time_diff = current_time - prev_time
        if time_diff < 0.001:  # Avoid division by zero
            return (0.0, 0.0)
            
        # Calculate pixel displacement
        pixel_dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        pixel_velocity = pixel_dist / time_diff
        
        # Calculate 3D displacement if we have valid depths
        if depth > 0 and prev_depth > 0:
            # This is a simplistic calculation, could be improved with camera intrinsics
            if self.camera_intrinsics:
                # More accurate 3D velocity calculation using camera parameters
                # Convert 2D points to 3D
                fx = self.camera_intrinsics['fx']
                fy = self.camera_intrinsics['fy']
                cx = self.camera_intrinsics['cx']
                cy = self.camera_intrinsics['cy']
                
                # Calculate 3D coordinates
                x1_3d = (prev_x - cx) * prev_depth / fx
                y1_3d = (prev_y - cy) * prev_depth / fy
                z1_3d = prev_depth
                
                x2_3d = (x - cx) * depth / fx
                y2_3d = (y - cy) * depth / fy
                z2_3d = depth
                
                # Calculate 3D distance
                dist_3d = np.sqrt((x2_3d - x1_3d)**2 + (y2_3d - y1_3d)**2 + (z2_3d - z1_3d)**2)
                meter_velocity = dist_3d / time_diff
            else:
                # Simpler approximation
                depth_diff = abs(depth - prev_depth)
                meter_velocity = depth_diff / time_diff
        else:
            meter_velocity = 0.0
            
        return (pixel_velocity, meter_velocity)

    def calculate_risk_score(self, depth, velocity, class_name, bbox_area, img_area):
        """
        Calculate risk score based on depth, velocity, class priority, and size.
        
        Formula: risk = w1 * (1.0 / depth) + w2 * velocity + w3 * class_priority + w4 * (bbox_area / image_area)
        
        Args:
            depth: Object depth in meters
            velocity: Object velocity (tuple of pixel_vel, meter_vel)
            class_name: Object class name
            bbox_area: Bounding box area in pixels
            img_area: Total image area in pixels
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        # Get class priority (default to 0.5 if not in dictionary)
        class_priority = self.class_priorities.get(class_name.lower(), 0.5)
        
        # Handle invalid depth
        if depth <= 0:
            depth_factor = 0.0
        else:
            # Normalize inverse depth to 0-1 range
            # Objects closer than min_depth get max risk (1.0)
            # Objects further than max_depth get min risk (0.0)
            depth_factor = min(1.0, max(0.0, (1.0 / depth - 1.0 / self.max_depth) / 
                                       (1.0 / self.min_depth - 1.0 / self.max_depth)))
        
        # Use meter velocity if available, otherwise pixel velocity
        pixel_vel, meter_vel = velocity
        if meter_vel > 0:
            # Normalize velocity (assumes max velocity of 5 m/s)
            vel_factor = min(1.0, meter_vel / 5.0)
        else:
            # Normalize pixel velocity (assumes max velocity of 500 pixels/s)
            vel_factor = min(1.0, pixel_vel / 500.0)
        
        # Normalize size factor
        size_factor = min(1.0, bbox_area / img_area)
        
        # Calculate weighted risk score
        risk_score = (self.risk_depth_weight * depth_factor + 
                     self.risk_velocity_weight * vel_factor + 
                     self.risk_class_weight * class_priority + 
                     self.risk_size_weight * size_factor)
        
        # Ensure risk is between 0 and 1
        risk_score = min(1.0, max(0.0, risk_score))
        
        return risk_score

    def track_objects(self, current_time, detections, depth_image, img_shape):
        """
        Track objects across frames using IoU matching.
        
        Args:
            current_time: Current timestamp
            detections: List of detections (class_name, box, depth, confidence)
            depth_image: Depth image
            img_shape: Image shape (height, width)
            
        Returns:
            Dictionary of tracked objects
        """
        img_area = img_shape[0] * img_shape[1]
        
        # Create boxes for current detections
        current_boxes = []
        current_data = []
        
        for class_name, (x, y, w, h), depth, confidence in detections:
            current_boxes.append((x, y, w, h))
            current_data.append((class_name, depth, confidence))
        
        # If no tracked objects yet, initialize with current detections
        if len(self.tracked_objects) == 0:
            for i, ((x, y, w, h), (class_name, depth, confidence)) in enumerate(zip(current_boxes, current_data)):
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Determine zone based on depth
                zone = self.determine_zone(depth)
                
                # Get color based on zone
                color = self.zone_colors[zone]
                
                # Create new tracked object
                obj = TrackedObject(
                    id=self.next_id,
                    class_name=class_name,
                    last_seen=current_time,
                    positions=deque(maxlen=self.tracking_history),
                    velocities=deque(maxlen=self.tracking_history),
                    risk_scores=deque(maxlen=self.tracking_history),
                    color=color,
                    zone=zone
                )
                
                # Add initial position
                obj.positions.append((current_time, center_x, center_y, w, h, depth))
                obj.velocities.append((0.0, 0.0))
                
                # Calculate risk score
                risk = self.calculate_risk_score(depth, (0.0, 0.0), class_name, w * h, img_area)
                obj.risk_scores.append(risk)
                
                # Add to tracked objects
                self.tracked_objects[self.next_id] = obj
                self.next_id += 1
                
            return self.tracked_objects
        
        # Calculate IoU between current detections and tracked objects
        iou_matrix = np.zeros((len(current_boxes), len(self.tracked_objects)))
        tracked_obj_ids = list(self.tracked_objects.keys())
        
        for i, current_box in enumerate(current_boxes):
            for j, obj_id in enumerate(tracked_obj_ids):
                obj = self.tracked_objects[obj_id]
                # Use the most recent position
                if len(obj.positions) > 0:
                    _, _, _, w, h, _ = obj.positions[-1]
                    # Check if object has predicted position
                    if obj.predicted_position is not None:
                        tracked_box = obj.predicted_position
                    else:
                        # Use last position
                        _, cx, cy, w, h, _ = obj.positions[-1]
                        tracked_box = (int(cx - w / 2), int(cy - h / 2), w, h)
                    
                    iou = self.calculate_iou(current_box, tracked_box)
                    iou_matrix[i, j] = iou
        
        # Find matches using greedy assignment
        matched_indices = []
        
        # Sort IoU matrix by IoU values (largest first)
        iou_sorted = np.argsort(-iou_matrix, axis=None)
        iou_flat = iou_matrix.flatten()
        
        # Perform matching
        assigned_rows = set()
        assigned_cols = set()
        
        for idx in range(len(iou_sorted)):
            # Get the indices
            flat_idx = iou_sorted[idx]
            if iou_flat[flat_idx] < self.iou_threshold:
                break  # No more matches above threshold
                
            # Convert flat index to 2D indices
            i = flat_idx // len(self.tracked_objects)
            j = flat_idx % len(self.tracked_objects)
            
            # Bounds checking (should never happen, but just in case)
            if i >= len(current_boxes) or j >= len(tracked_obj_ids):
                continue
                
            # Check if already assigned
            if i in assigned_rows or j in assigned_cols:
                continue
                
            matched_indices.append((i, j))
            assigned_rows.add(i)
            assigned_cols.add(j)
        
        # Process matched detections and update tracked objects
        for i, j in matched_indices:
            obj_id = tracked_obj_ids[j]
            obj = self.tracked_objects[obj_id]
            
            # Get current detection data
            x, y, w, h = current_boxes[i]
            class_name, depth, confidence = current_data[i]
            
            # Update class name if confidence is higher or same class
            if class_name == obj.class_name or confidence > 0.7:
                obj.class_name = class_name
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate velocity
            velocity = self.calculate_velocity(obj, current_time, center_x, center_y, depth)
            
            # Update zone based on depth
            obj.zone = self.determine_zone(depth)
            obj.color = self.zone_colors[obj.zone]
            
            # Update object state
            obj.last_seen = current_time
            obj.positions.append((current_time, center_x, center_y, w, h, depth))
            obj.velocities.append(velocity)
            obj.update_depth_stats()
            
            # Calculate risk score
            risk = self.calculate_risk_score(depth, velocity, obj.class_name, w * h, img_area)
            obj.risk_scores.append(risk)
            
            # Clear predicted position as we have an actual observation
            obj.predicted_position = None
        
        # Process unmatched detections (new objects)
        for i in range(len(current_boxes)):
            if i not in assigned_rows:
                x, y, w, h = current_boxes[i]
                class_name, depth, confidence = current_data[i]
                
                # Skip if not confident enough or depth is invalid
                if confidence < 0.3:
                    continue
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Determine zone based on depth
                zone = self.determine_zone(depth)
                
                # Get color based on zone
                color = self.zone_colors[zone]
                
                # Create new tracked object
                obj = TrackedObject(
                    id=self.next_id,
                    class_name=class_name,
                    last_seen=current_time,
                    positions=deque(maxlen=self.tracking_history),
                    velocities=deque(maxlen=self.tracking_history),
                    risk_scores=deque(maxlen=self.tracking_history),
                    color=color,
                    zone=zone
                )
                
                # Add initial position
                obj.positions.append((current_time, center_x, center_y, w, h, depth))
                obj.velocities.append((0.0, 0.0))
                
                # Calculate risk score
                risk = self.calculate_risk_score(depth, (0.0, 0.0), class_name, w * h, img_area)
                obj.risk_scores.append(risk)
                
                # Add to tracked objects
                self.tracked_objects[self.next_id] = obj
                self.next_id += 1
        
        # Process unmatched tracked objects (missing detections)
        for j, obj_id in enumerate(tracked_obj_ids):
            if j not in assigned_cols:
                obj = self.tracked_objects[obj_id]
                
                # Simple linear prediction for next position
                if len(obj.positions) >= 2:
                    # Get last two positions
                    t2, x2, y2, w, h, _ = obj.positions[-1]
                    t1, x1, y1, _, _, _ = obj.positions[-2]
                    
                    # Calculate time difference
                    dt = t2 - t1
                    if dt > 0:
                        # Calculate velocity
                        vx = (x2 - x1) / dt
                        vy = (y2 - y1) / dt
                        
                        # Predict next position
                        dt_pred = current_time - t2
                        pred_x = int(x2 + vx * dt_pred)
                        pred_y = int(y2 + vy * dt_pred)
                        
                        # Store predicted position
                        obj.predicted_position = (
                            int(pred_x - w / 2),
                            int(pred_y - h / 2),
                            w,
                            h
                        )
                
        # Remove objects that haven't been seen for too long
        current_obj_ids = list(self.tracked_objects.keys())
        for obj_id in current_obj_ids:
            obj = self.tracked_objects[obj_id]
            if current_time - obj.last_seen > self.max_tracking_age:
                del self.tracked_objects[obj_id]
                self.get_logger().debug(f"Removed tracked object {obj_id} ({obj.class_name}) due to age")
                
        return self.tracked_objects

    def draw_distance_zones(self, image, height):
        """
        Draw distance zone bands at the bottom of the image.
        
        Args:
            image: RGB image
            height: Height of the zone visualization bar
            
        Returns:
            Image with zone visualization
        """
        img_h, img_w = image.shape[:2]
        
        # Draw zone bands at the bottom of the image
        y_start = img_h - height
        y_end = img_h
        
        # Calculate width for each zone (scaled by distance thresholds)
        total_width = self.yellow_zone_threshold + (self.max_depth - self.yellow_zone_threshold) / 2
        
        red_width = int((self.red_zone_threshold / total_width) * img_w)
        yellow_width = int(((self.yellow_zone_threshold - self.red_zone_threshold) / total_width) * img_w)
        green_width = img_w - red_width - yellow_width
        
        # Create color bands
        cv2.rectangle(image, (0, y_start), (red_width, y_end), self.zone_colors["red"], -1)
        cv2.rectangle(image, (red_width, y_start), (red_width + yellow_width, y_end), self.zone_colors["yellow"], -1)
        cv2.rectangle(image, (red_width + yellow_width, y_start), (img_w, y_end), self.zone_colors["green"], -1)
        
        # Add zone labels
        cv2.putText(image, f"RED ZONE (<{self.red_zone_threshold}m)", (10, y_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(image, f"YELLOW ZONE (<{self.yellow_zone_threshold}m)", (red_width + 10, y_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.putText(image, f"GREEN ZONE (>{self.yellow_zone_threshold}m)", (red_width + yellow_width + 10, y_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image

    def draw_tracking_visualization(self, image, tracked_objects):
        """
        Draw tracking visualization overlays on the image.
        
        Args:
            image: RGB image
            tracked_objects: Dictionary of tracked objects
            
        Returns:
            Image with overlays
        """
        # First draw distance zone bands
        image = self.draw_distance_zones(image, height=40)
        
        # Process each tracked object
        for obj_id, obj in tracked_objects.items():
            if len(obj.positions) == 0:
                continue
                
            # Get the last position
            _, center_x, center_y, w, h, depth = obj.positions[-1]
            
            # Calculate bounding box coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            # Get velocity
            pixel_vel, meter_vel = obj.get_velocity()
            
            # Get color based on zone
            color = self.zone_colors[obj.zone]
            
            # If using predicted position, use dashed outline
            if obj.predicted_position is not None:
                x, y, w, h = obj.predicted_position
                # Draw dashed bounding box
                dash_length = 10
                for i in range(0, w, dash_length * 2):
                    x1 = x + i
                    x2 = min(x + i + dash_length, x + w)
                    cv2.line(image, (x1, y), (x2, y), color, 2)
                    cv2.line(image, (x1, y + h), (x2, y + h), color, 2)
                
                for i in range(0, h, dash_length * 2):
                    y1 = y + i
                    y2 = min(y + i + dash_length, y + h)
                    cv2.line(image, (x, y1), (x, y2), color, 2)
                    cv2.line(image, (x + w, y1), (x + w, y2), color, 2)
            else:
                # Draw solid bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Create label with all necessary information in one line
            label_parts = []
            label_parts.append(f"{obj.class_name}")
            
            if depth > 0:
                label_parts.append(f"{depth:.2f}m")
            
            if meter_vel > 0:
                label_parts.append(f"{meter_vel:.2f}m/s")
            
            label_parts.append(f"{obj.zone.upper()}")
            
            label = " | ".join(label_parts)
            
            # Draw background for text for better visibility
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x, y - text_size[1] - 10), (x + text_size[0], y), color, -1)
            
            # Text color based on zone for better visibility
            if obj.zone == "green":
                text_color = (0, 0, 0)  # Black text on green
            else:
                text_color = (255, 255, 255)  # White text on red/yellow
                
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Draw minimal trajectory (only last 5 positions)
            if len(obj.positions) >= 2:
                points = [(int(cx), int(cy)) for _, cx, cy, _, _, _ in list(obj.positions)[-5:]]
                for i in range(1, len(points)):
                    cv2.line(image, points[i-1], points[i], color, 2)
        
        # Draw compact global information in top-left corner
        cv2.rectangle(image, (10, 10), (250, 50), (0, 0, 0), -1)
        cv2.putText(image, f"Objects: {len(tracked_objects)}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

    def callback(self, rgb_msg, depth_msg, dets_msg: DetectionArray):
        """
        Callback function to process RGB image, depth image, and detections.
        Performs tracking and risk assessment with optimized performance.
        """
        try:
            # Convert ROS2 messages to OpenCV images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Get current time (ROS2 time in seconds)
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # Set frame time if first frame
            if self.last_frame_time is None:
                self.last_frame_time = current_time
            
            # Get image shape
            img_shape = rgb_image.shape[:2]  # (height, width)
            
            # Check if depth and RGB have different resolutions and adjust if needed
            if depth_image.shape[:2] != rgb_image.shape[:2]:
                depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
            
            # Process detections efficiently
            detections = []  # List of (class_name, (x, y, w, h), depth, confidence) tuples
            
            # Loop over all detections
            for det in dets_msg.detections:
                try:
                    box = det.bbox
                    
                    # Extract bounding box details
                    center_x = int(box.center.position.x)
                    center_y = int(box.center.position.y)
                    width = int(box.size.x)
                    height = int(box.size.y)
                    
                    # Skip invalid boxes
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Compute top-left corner coordinates of bounding box
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    w = int(width)
                    h = int(height)
                    
                    # Calculate depth using the improved method
                    depth_m, confidence = self.calculate_object_depth(depth_image, x, y, w, h)
                    
                    # Add to detections list
                    detections.append((det.class_name, (x, y, w, h), depth_m, confidence))
                    
                    # Log detection with zone information
                    zone = self.determine_zone(depth_m)
                    if depth_m > 0:
                        self.get_logger().info(f"Detected: {det.class_name}, Depth: {depth_m:.2f}m, Zone: {zone}")
                    
                except Exception as e:
                    self.get_logger().warn(f"Error processing detection: {str(e)}")
                    continue
            
            # Track objects
            tracked_objects = self.track_objects(current_time, detections, depth_image, img_shape)
            
            # Create visualization
            result_image = self.draw_tracking_visualization(rgb_image.copy(), tracked_objects)
            
            # Publish result
            out_msg = self.bridge.cv2_to_imgmsg(result_image, encoding='bgr8')
            out_msg.header = rgb_msg.header
            self.image_pub.publish(out_msg)
            
            # Update frame counter and time
            self.frame_counter += 1
            self.last_frame_time = current_time
            
            self.get_logger().info(f"Processed frame {self.frame_counter}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing data: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    """
    Main entry point for the ROS2 node. Initializes the node and starts processing.
    """
    rclpy.init(args=args)
    node = DistanceZoneTrackingNode()
    
    # Keep the node running until it's shutdown
    rclpy.spin(node)
    
    # Clean up when shutting down
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()