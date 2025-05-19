#!/usr/bin/env python3

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

@dataclass
class TrackedObject:
    """Class for storing tracked object information across frames"""
    id: int                          # Unique ID for this object
    class_name: str                  # YOLO class name
    first_detected: float            # Timestamp when first detected
    last_seen: float                 # Timestamp when last detected
    positions: deque = field(default_factory=lambda: deque(maxlen=20))  # Queue of (timestamp, x, y, w, h, depth) tuples
    velocities: deque = field(default_factory=lambda: deque(maxlen=20))  # Queue of velocity measurements (pixels/sec, m/sec)
    risk_scores: deque = field(default_factory=lambda: deque(maxlen=20))  # Queue of risk scores
    color: Tuple[int, int, int] = (0, 0, 0)  # Color for visualization
    avg_depth: float = 0.0           # Moving average of depth
    depth_variance: float = 0.0      # Variance in depth measurements
    predicted_position: Optional[Tuple[int, int, int, int]] = None  # Predicted next position
    zone: str = "unknown"            # Current zone based on depth
    
    def update_depth_stats(self):
        """Update moving average depth and variance using efficient NumPy operations"""
        depths = np.array([pos[5] for pos in self.positions if pos[5] > 0])
        if len(depths) > 0:
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

def calculate_iou(box1, box2):
    """
    Calculate intersection over union between two bounding boxes.
    Optimized for speed with direct array operations.
    
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
    
    # Calculate IoU with epsilon to prevent division by zero
    iou = intersection_area / max(union_area, 1e-6)
    
    return iou