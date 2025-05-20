#!/usr/bin/env python3

# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
from typing import List, Dict
from cv_bridge import CvBridge
import time

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
# ROS 2 Foxy compatible imports
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle.node import LifecycleState  # Changed path for Foxy

import torch
# Ultralytics imports with error handling
try:
    from ultralytics import YOLO, YOLOWorld
except ImportError:
    print("Warning: Ultralytics not installed, using pip to install...")
    import os
    os.system("pip install ultralytics")
    from ultralytics import YOLO, YOLOWorld

from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

# Import adapters for YOLOv6 and YOLOv7
from .adapters import YOLOv6, YOLOv7

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses


class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        # params
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)

        # Extended type_to_model to include all YOLO versions
        self.type_to_model = {
            "YOLO": YOLO,           # YOLOv5/v8 from Ultralytics
            "World": YOLOWorld,     # YOLOWorld from Ultralytics
            "YOLOv6": YOLOv6,       # YOLOv6 adapter
            "YOLOv7": YOLOv7        # YOLOv7 adapter
        }
        
        # Detection timing tracking
        self.object_last_detection = {}
        self.detection_time_ms = {}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # model params
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )

        # inference params
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )

        # ros params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )

        # detection pub
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, "detections", 10)
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            # Load model using the appropriate handler based on model_type
            model_class = self.type_to_model.get(self.model_type)
            if model_class is None:
                self.get_logger().error(f"Unsupported model type: {self.model_type}")
                return TransitionCallbackReturn.ERROR
                
            self.yolo = model_class(self.model)
            
            # Log model information
            self.get_logger().info(f"Loaded {self.model_type} model: {self.model}")
            if hasattr(self.yolo, 'device'):
                self.get_logger().info(f"Model device: {self.yolo.device}")
            
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist")
            return TransitionCallbackReturn.ERROR
        except Exception as e:
            self.get_logger().error(f"Error loading model: {str(e)}")
            return TransitionCallbackReturn.ERROR

        try:
            self.get_logger().info("Trying to fuse model...")
            self.yolo.fuse()
        except Exception as e:
            self.get_logger().warn(f"Error while fuse: {str(e)}")

        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        # Create set_classes service for all models
        self._set_classes_srv = self.create_service(
            SetClasses, "set_classes", self.set_classes_cb
        )

        self._sub = self.create_subscription(
            Image, "image_raw", self.image_cb, self.image_qos_profile
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        if hasattr(self, '_set_classes_srv') and self._set_classes_srv:
            self.destroy_service(self._set_classes_srv)
            self._set_classes_srv = None

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response
        
    def get_detection_time(self, obj_id, class_name):
        """Get detection time in milliseconds"""
        now = time.time() * 1000  # Current time in ms
        unique_id = f"{class_name}_{obj_id}"
        
        if unique_id not in self.object_last_detection:
            # First detection
            self.object_last_detection[unique_id] = now
            self.detection_time_ms[unique_id] = 0
            return 0
            
        # Calculate time since last detection
        time_diff = now - self.object_last_detection[unique_id]
        
        # If gap is significant (object was lost and now found)
        if time_diff > 100:  # 100ms threshold for redetection
            self.detection_time_ms[unique_id] = int(time_diff)
        else:
            # Continuous detection, no significant gap
            self.detection_time_ms[unique_id] = 0
            
        # Update last detection time
        self.object_last_detection[unique_id] = now
        
        return self.detection_time_ms[unique_id]

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        """Parse detection hypothesis."""
        hypothesis_list = []

        # Handle results differently based on model type
        if self.model_type in ["YOLOv6", "YOLOv7"]:
            # For adapter models, parse raw predictions
            for i, detection in enumerate(results._pred):
                if len(detection) >= 6:  # xmin, ymin, xmax, ymax, conf, cls
                    cls_id = int(detection[5])
                    hypothesis = {
                        "class_id": cls_id,
                        "class_name": self.yolo.names[cls_id],
                        "score": float(detection[4]),
                    }
                    hypothesis_list.append(hypothesis)
            return hypothesis_list
        
        # Standard Ultralytics format (YOLOv5/YOLOv8)
        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        """Parse bounding boxes."""
        boxes_list = []

        # Handle results differently based on model type
        if self.model_type in ["YOLOv6", "YOLOv7"]:
            # For adapter models, parse raw predictions
            for detection in results._pred:
                if len(detection) >= 6:  # xmin, ymin, xmax, ymax, conf, cls
                    msg = BoundingBox2D()
                    
                    # Get box in xyxy format
                    x1, y1, x2, y2 = detection[:4]
                    
                    # Convert to center format (xywh)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Set message fields
                    msg.center.position.x = float(center_x)
                    msg.center.position.y = float(center_y)
                    msg.center.theta = 0.0  # No rotation in YOLOv6/v7
                    msg.size.x = float(width)
                    msg.size.y = float(height)
                    
                    boxes_list.append(msg)
            return boxes_list
        
        # Standard Ultralytics format (YOLOv5/YOLOv8)
        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                msg = BoundingBox2D()

                # get boxes values
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()

                # get boxes values
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        """Parse segmentation masks."""
        masks_list = []
        
        # YOLOv6/v7 don't typically return masks
        if self.model_type in ["YOLOv6", "YOLOv7"]:
            return masks_list

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:
            msg = Mask()

            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        """Parse keypoints."""
        keypoints_list = []
        
        # YOLOv6/v7 don't typically return keypoints
        if self.model_type in ["YOLOv6", "YOLOv7"]:
            return keypoints_list

        points: Keypoints
        for points in results.keypoints:
            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def image_cb(self, msg: Image) -> None:
        if self.enable:
            # Start timing the detection process
            start_time = time.time()
            
            # Convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding=self.yolo_encoding
            )
            
            # Run prediction
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                half=self.half,
                max_det=self.max_det,
                augment=self.augment,
                agnostic_nms=self.agnostic_nms,
                retina_masks=self.retina_masks,
                device=self.device,
            )
            
            # For Ultralytics models
            if self.model_type in ["YOLO", "World"]:
                results = results[0].cpu()
            else:
                # For adapter models, results format is already correct
                results = results[0]
            
            # Check if we have any detections
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)
            
            # Standard Ultralytics models may have masks and keypoints
            masks = self.parse_masks(results) if hasattr(results, 'masks') and results.masks else []
            keypoints = self.parse_keypoints(results) if hasattr(results, 'keypoints') and results.keypoints else []

            # Create detection msgs
            detections_msg = DetectionArray()
            
            for i in range(len(hypothesis)):
                aux_msg = Detection()
                
                # Set detection fields
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                
                # Set bounding box if available
                if i < len(boxes):
                    aux_msg.bbox = boxes[i]
                
                # Track detection time for this object
                detection_ms = self.get_detection_time(i, aux_msg.class_name)
                aux_msg.detection_time_ms = int(detection_ms)
                
                # Set mask if available
                if masks and i < len(masks):
                    aux_msg.mask = masks[i]
                
                # Set keypoints if available
                if keypoints and i < len(keypoints):
                    aux_msg.keypoints = keypoints[i]
                
                detections_msg.detections.append(aux_msg)
            
            # Publish detections
            detections_msg.header = msg.header
            self._pub.publish(detections_msg)
            
            # Log performance
            elapsed = time.time() - start_time
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    self.get_logger().info(f"Inference time: {elapsed*1000:.1f}ms ({1.0/elapsed:.1f} FPS)")
            
            # Clean up
            del results
            del cv_image

    def set_classes_cb(
        self,
        req: SetClasses.Request,
        res: SetClasses.Response,
    ) -> SetClasses.Response:
        """Set classes callback."""
        self.get_logger().info(f"Setting classes: {req.classes}")
        try:
            # Set classes differently based on model type
            if hasattr(self.yolo, "set_classes"):
                self.yolo.set_classes(req.classes)
                self.get_logger().info(f"Classes set successfully")
            else:
                self.get_logger().warn(f"Model doesn't support setting classes")
        except Exception as e:
            self.get_logger().error(f"Error setting classes: {str(e)}")
        return res


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()