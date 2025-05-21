import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    "yolo_bringup"), "launch", "yolo.launch.py")),
            launch_arguments={
                "model": LaunchConfiguration("model", default="yolov8n.pt"),
                "tracker": LaunchConfiguration("tracker", default="bytetrack.yaml"),
                "device": LaunchConfiguration("device", default="cuda:0"),
                "enable": LaunchConfiguration("enable", default="True"),
                "threshold": LaunchConfiguration("threshold", default="0.5"),
                "input_image_topic": LaunchConfiguration("input_image_topic", default="/camera/camera/color/image_raw"),
                "image_reliability": LaunchConfiguration("image_reliability", default="2"),
                "namespace": LaunchConfiguration("namespace", default="yolo"),
            }.items(),
        )
    ])
