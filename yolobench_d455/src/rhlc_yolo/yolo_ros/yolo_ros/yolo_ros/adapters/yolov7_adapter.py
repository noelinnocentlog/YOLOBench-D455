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

class ResultsYOLOv7:
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

class YOLOv7:
    """Adapter class for YOLOv7 that mimics the Ultralytics YOLO API."""
    
    def __init__(self, model_path):
        """
        Initialize YOLOv7 model.
        
        Args:
            model_path: Path to YOLOv7 model weights
        """
        # Add YOLOv7 directory to path (will be downloaded during first use)
        yolov7_dir = os.path.join(os.path.expanduser('~'), '.yolov7')
        os.makedirs(yolov7_dir, exist_ok=True)
        sys.path.append(yolov7_dir)
        
        # Clone YOLOv7 repo if not present
        if not os.path.exists(os.path.join(yolov7_dir, 'models')):
            print(f"Cloning YOLOv7 repository to {yolov7_dir}...")
            os.system(f"git clone https://github.com/WongKinYiu/yolov7.git {yolov7_dir}")
        
        # Import YOLOv7 modules
        try:
            # Use the main repo directory
            sys.path.append(yolov7_dir)
            # Import needed modules
            from models.experimental import attempt_load
            from utils.general import non_max_suppression, check_img_size
            from utils.torch_utils import select_device
            from utils.augmentations import letterbox
        except ImportError as e:
            print(f"Error importing YOLOv7 modules: {e}")
            print("Installing dependencies...")
            os.system(f"cd {yolov7_dir} && pip install -r requirements.txt")
            # Try import again
            from models.experimental import attempt_load
            from utils.general import non_max_suppression, check_img_size
            from utils.torch_utils import select_device
            from utils.augmentations import letterbox
            
        self.model_path = model_path
        self.device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = False  # Will be set during predict()
        
        # Load model
        self.model = attempt_load(model_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(640, s=self.stride)
        
        # Save functions for later use
        self.non_max_suppression = non_max_suppression
        self.letterbox = letterbox
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Class names from model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
    def fuse(self):
        """Fuse model Conv2d and BatchNorm2d layers for faster inference."""
        try:
            from utils.torch_utils import fuse_conv_and_bn
            for m in self.model.modules():
                if type(m) in [torch.nn.Conv2d, torch.nn.BatchNorm2d]:
                    m.eval()
                if type(m) is torch.nn.Conv2d and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.fuseforward
            return True
        except Exception as e:
            print(f"Error fusing YOLOv7 model: {e}")
            return False
        
    def predict(self, source, conf=0.25, iou=0.45, max_det=300, 
               imgsz=(640, 640), half=False, augment=False, 
               agnostic_nms=False, retina_masks=False, device=None, **kwargs):
        """
        Run YOLOv7 inference on an image.
        
        Args:
            source: Input image (numpy array)
            conf: Confidence threshold
            iou: IoU threshold
            max_det: Maximum detections
            imgsz: Input image size
            half: Use half precision (FP16)
            augment: Augmented inference
            agnostic_nms: Class-agnostic NMS
            retina_masks: High-resolution masks (not supported in YOLOv7)
            device: Device to run on (cuda device, i.e. 0 or 0,1,2,3 or cpu)
            
        Returns:
            Results object with detections
        """
        # Set device
        if device:
            if 'cuda' in device:
                dev = 'cuda'
            else:
                dev = 'cpu'
            self.device = select_device(dev)
            self.model.to(self.device)
            
        # Set half precision
        self.half = half
        if half and self.device.type != 'cpu':
            self.model.half()
        
        # Store original image
        orig_img = source.copy()
        
        # Prepare image
        img = self.letterbox(source, imgsz[0], stride=self.stride)[0]
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
            pred = self.model(img, augment=augment)[0]
            
            # Apply NMS
            pred = self.non_max_suppression(
                pred, 
                conf_thres=conf, 
                iou_thres=iou,
                max_det=max_det,
                agnostic=agnostic_nms,
                classes=None
            )
            
        # Convert to numpy for Results object
        if len(pred) > 0:
            pred = pred[0].cpu().numpy()  # First batch item
        else:
            pred = np.zeros((0, 6))  # Empty array with 6 columns
            
        # Create Results object
        results = ResultsYOLOv7(pred, orig_img, self.names)
        
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