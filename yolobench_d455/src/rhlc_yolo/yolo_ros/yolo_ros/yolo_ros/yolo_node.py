#!/usr/bin/env python3
# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modified for Jetson AGX Xavier compatibility

import cv2
from typing import List, Dict
from cv_bridge import CvBridge
import time
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

import torch
from ultralytics import YOLO, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

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


class YoloNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_node")
        
        # Set environment variables for Jetson Xavier performance
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Optimize CUDA operations for Xavier
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        
        # Params with Xavier-optimized defaults
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8n.pt")  # Smaller model by default for Xavier
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # Xavier-optimized inference parameters
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 416)  # Smaller default size for Xavier
        self.declare_parameter("imgsz_width", 416)   # Smaller default size for Xavier
        self.declare_parameter("half", True)  # Default to half precision for Xavier
        self.declare_parameter("max_det", 100)  # Reduced detection limit for performance
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        
        # TensorRT optimization for Xavier
        self.declare_parameter("use_tensorrt", False)  # Disable by default initially
        self.declare_parameter("tensorrt_fp16", True)  # Use FP16 precision for TensorRT

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld}
        
        # Detection timing tracking
        self.object_last_detection = {}
        self.detection_time_ms = {}
        
        # Performance monitoring
        self.frame_count = 0
        self.total_inference_time = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        # Initialize parameters
        self.init_parameters()
        
        # Initialize publishers & subscribers
        self.init_communications()
        
        # Load model
        self.load_model()
        
        self.get_logger().info("YoloNode initialized successfully")

    def init_parameters(self):
        # Model params
        self.model_type = self.get_parameter("model_type").get_parameter_value().string_value
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.yolo_encoding = self.get_parameter("yolo_encoding").get_parameter_value().string_value

        # Inference params
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.retina_masks = self.get_parameter("retina_masks").get_parameter_value().bool_value
        
        # TensorRT parameters
        self.use_tensorrt = self.get_parameter("use_tensorrt").get_parameter_value().bool_value
        self.tensorrt_fp16 = self.get_parameter("tensorrt_fp16").get_parameter_value().bool_value

        # ROS params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

    def init_communications(self):
        # Detection pub
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Create publisher
        self._pub = self.create_publisher(DetectionArray, "detections", 10)
        
        # Create subscription 
        self._sub = self.create_subscription(
            Image, "image_raw", self.image_cb, self.image_qos_profile
        )
        
        # Performance publisher
        self._perf_pub = self.create_publisher(Image, "performance_overlay", 1)
        
        # Services
        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)
        
        self.cv_bridge = CvBridge()

    def load_model(self):
        # Xavier GPU setup
        try:
            # Free CUDA memory
            torch.cuda.empty_cache()
            
            # Check Xavier GPU capabilities
            if torch.cuda.is_available():
                self.get_logger().info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                self.get_logger().info(f"CUDA version: {torch.version.cuda}")
                self.get_logger().info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                self.get_logger().warn("CUDA not available, falling back to CPU")
                self.device = "cpu"
                
        except Exception as e:
            self.get_logger().error(f"GPU setup error: {str(e)}")

        try:
            # Create model with TensorRT optimization if enabled
            if self.use_tensorrt and "cuda" in self.device:
                self.get_logger().info("Loading model with TensorRT optimization...")
                # Initialize YOLO model first
                model_cls = self.type_to_model[self.model_type]
                self.yolo = model_cls(self.model)
                
                # Export to TensorRT if needed
                try:
                    trt_model_path = f"{self.model.split('.')[0]}_tensorrt_fp{'16' if self.tensorrt_fp16 else '32'}.engine"
                    if not os.path.exists(trt_model_path):
                        self.get_logger().info(f"Exporting model to TensorRT: {trt_model_path}")
                        self.yolo.export(format="engine", half=self.tensorrt_fp16, imgsz=[self.imgsz_height, self.imgsz_width])
                    
                    # Reload with TensorRT engine
                    self.yolo = model_cls(trt_model_path)
                    self.get_logger().info("Successfully loaded TensorRT optimized model")
                except Exception as e:
                    self.get_logger().error(f"TensorRT conversion failed: {str(e)}")
                    self.get_logger().warn("Falling back to regular CUDA model")
                    self.yolo = model_cls(self.model)
            else:
                self.yolo = self.type_to_model[self.model_type](self.model)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist")
            raise
        except Exception as e:
            self.get_logger().error(f"Model loading error: {str(e)}")
            raise

        try:
            if not self.use_tensorrt:  # Skip if using TensorRT
                self.get_logger().info("Trying to fuse model...")
                self.yolo.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fuse: {e}")
        except Exception as e:
            self.get_logger().warn(f"Unknown error while fuse: {e}")

        # Warm up the model to initialize CUDA context
        if "cuda" in self.device:
            self.get_logger().info("Warming up model...")
            dummy_input = torch.zeros((1, 3, self.imgsz_height, self.imgsz_width), device=self.device)
            try:
                # Run inference on dummy data for warmup
                for _ in range(2):
                    if not self.use_tensorrt:
                        self.yolo.predict(source=dummy_input, verbose=False)
                    # If using TensorRT, the first real prediction will warmup
            except Exception as e:
                self.get_logger().warn(f"Warmup error (non-critical): {str(e)}")
                
        if isinstance(self.yolo, YOLOWorld):
            self._set_classes_srv = self.create_service(
                SetClasses, "set_classes", self.set_classes_cb
            )

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
        # More robust error handling for Xavier
        try:
            hypothesis_list = []

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
        except Exception as e:
            self.get_logger().error(f"Error in parse_hypothesis: {str(e)}")
            return []

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        try:
            boxes_list = []

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
        except Exception as e:
            self.get_logger().error(f"Error in parse_boxes: {str(e)}")
            return []

    def parse_masks(self, results: Results) -> List[Mask]:
        try:
            masks_list = []

            def create_point2d(x: float, y: float) -> Point2D:
                p = Point2D()
                p.x = x
                p.y = y
                return p

            mask: Masks
            for mask in results.masks:
                msg = Mask()

                # Optimize for Xavier: limit the number of points for performance
                if mask.xy[0].shape[0] > 100:  # If more than 100 points
                    # Sample points to reduce computation
                    step = max(1, mask.xy[0].shape[0] // 100)
                    sampled_points = mask.xy[0][::step]
                    msg.data = [
                        create_point2d(float(ele[0]), float(ele[1]))
                        for ele in sampled_points.tolist()
                    ]
                else:
                    msg.data = [
                        create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()
                    ]
                
                msg.height = results.orig_img.shape[0]
                msg.width = results.orig_img.shape[1]

                masks_list.append(msg)

            return masks_list
        except Exception as e:
            self.get_logger().error(f"Error in parse_masks: {str(e)}")
            return []

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        try:
            keypoints_list = []

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
        except Exception as e:
            self.get_logger().error(f"Error in parse_keypoints: {str(e)}")
            return []

    def create_performance_overlay(self, cv_image, inference_time):
        """Create performance statistics overlay"""
        # Calculate and update FPS
        self.frame_count += 1
        self.total_inference_time += inference_time
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update > 1.0:
            elapsed = current_time - self.last_fps_update
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
            
        # Create copy for overlay
        overlay = cv_image.copy()
        
        # Draw performance stats
        avg_inference = self.total_inference_time / max(1, self.frame_count) * 1000
        
        cv2.putText(overlay, f"FPS: {self.fps:.1f}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add Xavier GPU stats if available
        try:
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(0) / 1e6  # MB
                mem_reserved = torch.cuda.memory_reserved(0) / 1e6    # MB
                
                cv2.putText(overlay, f"GPU Mem: {mem_allocated:.0f}/{mem_reserved:.0f} MB", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except:
            pass
            
        return overlay

    def image_cb(self, msg: Image) -> None:
        if self.enable:
            # Start timing the detection process
            start_time = time.time()
            
            try:
                # Convert image to tensor directly if possible for faster processing
                cv_image = self.cv_bridge.imgmsg_to_cv2(
                    msg, desired_encoding=self.yolo_encoding
                )
                
                # Run inference
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
                
                # Measure inference time
                inference_time = time.time() - start_time
                
                # Post-process results
                results: Results = results[0].cpu()  # Move to CPU for post-processing
                
                hypothesis = []
                boxes = []
                masks = []
                keypoints = []
                
                if results.boxes or results.obb:
                    hypothesis = self.parse_hypothesis(results)
                    boxes = self.parse_boxes(results)

                if results.masks:
                    masks = self.parse_masks(results)

                if results.keypoints:
                    keypoints = self.parse_keypoints(results)

                # Create detection messages
                detections_msg = DetectionArray()

                for i in range(len(results)):
                    aux_msg = Detection()

                    if (results.boxes or results.obb) and hypothesis and boxes and i < len(hypothesis) and i < len(boxes):
                        aux_msg.class_id = hypothesis[i]["class_id"]
                        aux_msg.class_name = hypothesis[i]["class_name"]
                        aux_msg.score = hypothesis[i]["score"]
                        aux_msg.bbox = boxes[i]
                        
                        # Track detection time for this object
                        detection_ms = self.get_detection_time(i, aux_msg.class_name)
                        
                        # Add detection time to the message
                        aux_msg.detection_time_ms = int(detection_ms)

                    if results.masks and masks and i < len(masks):
                        aux_msg.mask = masks[i]

                    if results.keypoints and keypoints and i < len(keypoints):
                        aux_msg.keypoints = keypoints[i]

                    detections_msg.detections.append(aux_msg)

                # Publish detections
                detections_msg.header = msg.header
                self._pub.publish(detections_msg)
                
                # Publish performance overlay
                try:
                    perf_overlay = self.create_performance_overlay(cv_image, inference_time)
                    perf_msg = self.cv_bridge.cv2_to_imgmsg(perf_overlay, encoding="bgr8")
                    perf_msg.header = msg.header
                    self._perf_pub.publish(perf_msg)
                except Exception as e:
                    self.get_logger().debug(f"Error creating performance overlay: {str(e)}")
                
                # Log performance occasionally
                if self.frame_count % 100 == 0:
                    self.get_logger().info(
                        f"Performance: {self.fps:.1f} FPS, Inference: {inference_time*1000:.1f}ms, "
                        f"Objects: {len(detections_msg.detections)}"
                    )
            
            except Exception as e:
                self.get_logger().error(f"Error in image_cb: {str(e)}")
            
            finally:
                # Force garbage collection to free memory
                try:
                    if "cuda" in self.device and self.frame_count % 30 == 0:
                        torch.cuda.empty_cache()
                except:
                    pass

    def set_classes_cb(
        self,
        req: SetClasses.Request,
        res: SetClasses.Response,
    ) -> SetClasses.Response:
        self.get_logger().info(f"Setting classes: {req.classes}")
        try:
            self.yolo.set_classes(req.classes)
            self.get_logger().info(f"New classes: {self.yolo.names}")
            res.success = True
        except Exception as e:
            self.get_logger().error(f"Error setting classes: {str(e)}")
            res.success = False
            res.message = str(e)
            
        return res


def main():
    rclpy.init()
    
    try:
        node = YoloNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()