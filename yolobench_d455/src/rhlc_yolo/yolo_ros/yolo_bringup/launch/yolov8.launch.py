#!/usr/bin/env python3
# Launch file optimized for Jetson AGX Xavier with Foxy

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Xavier-optimized parameters
    model_cmd = DeclareLaunchArgument(
        "model",
        default_value="yolov8n.pt", 
        description="YOLO model (n=smallest/fastest)"
    )
    
    imgsz_height_cmd = DeclareLaunchArgument(
        "imgsz_height",
        default_value="384",
        description="Image height for inference"
    )
    
    imgsz_width_cmd = DeclareLaunchArgument(
        "imgsz_width",
        default_value="384",
        description="Image width for inference"
    )
    
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.45",
        description="Detection confidence threshold"
    )
    
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device (cuda:0 or cpu)"
    )
    
    half_cmd = DeclareLaunchArgument(
        "half",
        default_value="True",
        description="Use half precision (FP16)"
    )
    
    tracker_cmd = DeclareLaunchArgument(
        "tracker",
        default_value="bytetrack.yaml",
        description="Tracker config file"
    )
    
    # Main detection node
    yolo_node = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_node",
        namespace="yolo",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "imgsz_height": LaunchConfiguration("imgsz_height"),
            "imgsz_width": LaunchConfiguration("imgsz_width"),
            "threshold": LaunchConfiguration("threshold"),
            "device": LaunchConfiguration("device"),
            "half": LaunchConfiguration("half"),
            "max_det": 100,  # Reduced for performance
            "enable": True,
            "performance_overlay": True,
        }],
        remappings=[("image_raw", "/camera/camera/color/image_raw")],
    )
    
    # Tracking node
    tracking_node = Node(
        package="yolo_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace="yolo",
        parameters=[{
            "tracker": LaunchConfiguration("tracker"),
            "process_every_n_frame": 1,
            "tracker_fps": 15.0,
            "downscale_factor": 1.0,
            "performance_monitor": True,
            "max_objects": 20,
            "use_cuda": True,
        }],
        remappings=[("image_raw", "/camera/camera/color/image_raw")],
    )
    
    # Debug visualization node
    debug_node = Node(
        package="yolo_ros",
        executable="debug_node",
        name="debug_node",
        namespace="yolo",
        parameters=[{
            "enable_markers": True,
            "enable_keypoints": True,
            "enable_masks": True,
            "text_scale": 0.6,
            "line_thickness": 1,
            "performance_overlay": True,
        }],
        remappings=[
            ("image_raw", "/camera/camera/color/image_raw"),
            ("detections", "tracking"),
        ],
    )
    
    return LaunchDescription([
        model_cmd,
        imgsz_height_cmd,
        imgsz_width_cmd,
        threshold_cmd,
        device_cmd,
        half_cmd,
        tracker_cmd,
        yolo_node,
        tracking_node,
        debug_node,
    ])