#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    package_name = 'yolo_bench_d455'
    
    # Paths
    rviz_config_path = os.path.join(get_package_share_directory(package_name), 'rviz', 'yolo_bench_d455.rviz')
    
    # Launch arguments
    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')
    
    # Declare arguments
    arg_declarations = [
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(get_package_share_directory(package_name), 'config', 'default_params.yaml'),
            description='Path to parameters file'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='True',
            description='Whether to launch RViz'
        ),
    ]
    
    # Include the tracking launch file
    tracking_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package_name), 'launch', 'tracking_launch.py')
        ),
        launch_arguments={
            'params_file': params_file,
        }.items()
    )
    
    # Create visualization node
    visualization_node = Node(
        name='tracking_visualization',
        namespace='',
        package='yolo_bench_d455',
        node_executable='visualization',
        parameters=[{
            'tracking_topic': '/yolo/depth_tracking_image',
            'publish_point_cloud': True,
        }],
        output='screen'
    )
    
    # Create RViz node
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    return LaunchDescription(arg_declarations + [
        tracking_launch,
        visualization_node,
        rviz_node,
    ])