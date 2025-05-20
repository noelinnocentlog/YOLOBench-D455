# File: yolov7.launch.py
#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchArgument, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription(
        [
            # Declare arguments with default values
            DeclareLaunchArgument(
                'model',
                default_value='yolov7.pt',
                description='YOLOv7 model path'
            ),
            
            DeclareLaunchArgument(
                'device', 
                default_value='cuda:0',
                description='Device to run inference on (cuda:0 or cpu)'
            ),
            
            DeclareLaunchArgument(
                'threshold',
                default_value='0.5',
                description='Detection confidence threshold'
            ),
            
            DeclareLaunchArgument(
                'input_image_topic',
                default_value='/camera/color/image_raw',
                description='Input image topic'
            ),
            
            # Include the base launch file
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("yolo_bringup"),
                        "launch",
                        "yolo.launch.py",
                    )
                ),
                launch_arguments={
                    "model_version": "yolov7",
                    "model": LaunchConfiguration("model"),
                    "tracker": "bytetrack.yaml",
                    "device": LaunchConfiguration("device"),
                    "enable": "True",
                    "half": "True",  # Use half precision for Jetson
                    "imgsz_height": "640",  # Good balance for Jetson
                    "imgsz_width": "640",
                    "threshold": LaunchConfiguration("threshold"),
                    "input_image_topic": LaunchConfiguration("input_image_topic"),
                    "image_reliability": "1",
                    "namespace": "yolo",
                }.items(),
            )
        ]
    )