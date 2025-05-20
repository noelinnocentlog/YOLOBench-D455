#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Any, Optional, Union

# Results class to mimic Ultralytics API
class BoxResults:
    def __init__(self, boxes, cls, conf, orig_img):
        self.boxes = boxes
        self.orig_img = orig_img
        
    @property
    def cpu(self):
        return self

class ResultsYOLOv6:
    def __init__(self, pred, orig_img, names):
        self.names = names
        self.orig_img = orig_img
        self.boxes = BoxResults(True, None, None, orig_img) if len(pred) > 0 else None
        self.obb = None
        self.masks = None
        self.keypoints = None
        self._pred = pred  # Store the raw predictions
        
    def __getitem__(self, idx):
        return self
        
    def __len__(self):
        return len(self._pred)

class YOLOv6:
    """Adapter class for YOLOv6 that mimics the Ultralytics YOLO API."""
    
    def __init__(self, model_path):
        """
        Initialize YOLOv6 model.
        
        Args:
            model_path: Path to YOLOv6 model weights
        """
        # Add YOLOv6 directory to path (will be downloaded during first use)
        yolov6_dir = os.path.join(os.path.expanduser('~'), '.yolov6')
        os.makedirs(yolov6_dir, exist_ok=True)
        sys.path.append(yolov6_dir)
        
        # Clone YOLOv6 repo if not present
        if not os.path.exists(os.path.join(yolov6_dir, 'tools')):
            print(f"Cloning YOLOv6 repository to {yolov6_dir}...")
            os.system(f"git clone https://github.com/meituan/YOLOv6.git {yolov6_dir}")
        
        # Import YOLOv6 modules
        try:
            # Try import with error handling
            from tools.infer import Inferer
            from configs.default_config import get_config
            from yolov6.utils.checkpoint import load_checkpoint
            from yolov6.layers.common import fuse_conv_bn
            from yolov6.models.yolo import build_model
            from yolov6.utils.nms import non_max_suppression
        except ImportError as e:
            print(f"Error importing YOLOv6 modules: {e}")
            print("Installing dependencies...")
            os.system(f"cd {yolov6_dir} && pip install -r requirements.txt")
            # Try import again
            from tools.infer import Inferer
            from configs.default_config import get_config
            from yolov6.utils.checkpoint import load_checkpoint
            from yolov6.layers.common import fuse_conv_bn
            from yolov6.models.yolo import build_model
            from yolov6.utils.nms import non_max_suppression
            
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = False  # Will be set during predict()
        
        # Setup config
        self.cfg = get_config()
        self.cfg.model = model_path
        self.cfg.batch_size = 1
        self.cfg.img_size = 640  # Default
        self.cfg.conf_thres = 0.25  # Default
        self.cfg.iou_thres = 0.45  # Default
        self.cfg.max_det = 1000  # Default
        
        # Load model
        self.model = build_model(self.cfg)
        self.model = load_checkpoint(self.model, self.model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Store non_max_suppression for use in predict()
        self.non_max_suppression = non_max_suppression
        
        # Class names (COCO by default)
        self.names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
    def fuse(self):
        """Fuse model Conv2d and BatchNorm2d layers for faster inference."""
        try:
            from yolov6.layers.common import fuse_model
            self.model = fuse_model(self.model)
            return True
        except Exception as e:
            print(f"Error fusing YOLOv6 model: {e}")
            return False
        
    def predict(self, source, conf=0.25, iou=0.45, max_det=300, 
               imgsz=(640, 640), half=False, augment=False, 
               agnostic_nms=False, retina_masks=False, device=None, **kwargs):
        """
        Run YOLOv6 inference on an image.
        
        Args:
            source: Input image (numpy array)
            conf: Confidence threshold
            iou: IoU threshold
            max_det: Maximum detections
            imgsz: Input image size
            half: Use half precision (FP16)
            augment: Augmented inference
            agnostic_nms: Class-agnostic NMS
            retina_masks: High-resolution masks (not supported in YOLOv6)
            device: Device to run on (cuda device, i.e. 0 or 0,1,2,3 or cpu)
            
        Returns:
            Results object with detections
        """
        # Update config
        self.cfg.conf_thres = conf
        self.cfg.iou_thres = iou
        self.cfg.max_det = max_det
        self.cfg.img_size = imgsz
        
        # Set device
        if device:
            self.device = device
            self.model.to(self.device)
            
        # Set half precision
        self.half = half
        if half and self.device != 'cpu':
            self.model.half()
        
        # Store original image
        orig_img = source.copy()
        
        # Prepare image
        img = cv2.resize(source, (imgsz[1], imgsz[0]))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Some YOLOv6 models return tuple
                
            # Apply NMS
            pred = self.non_max_suppression(
                outputs, 
                conf_thres=conf, 
                iou_thres=iou,
                max_det=max_det,
                agnostic=agnostic_nms
            )
            
        # Convert to Ultralytics format
        if len(pred) > 0:
            pred = pred[0].cpu().numpy()  # First batch item
        else:
            pred = np.zeros((0, 6))  # Empty array with 6 columns
            
        # Create Results object
        results = ResultsYOLOv6(pred, orig_img, self.names)
        
        return [results]
    
    def set_classes(self, classes):
        """
        Set classes for the model (to mimic YOLOWorld behavior).
        
        Args:
            classes: List of class names
        """
        # Create new mapping of indices to class names
        self.names = {i: name for i, name in enumerate(classes)}
        return True