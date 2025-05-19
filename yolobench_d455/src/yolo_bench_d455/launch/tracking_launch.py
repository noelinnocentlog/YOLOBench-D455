#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    package_name = 'yolo_bench_d455'
    
    # Paths
    config_dir = os.path.join(get_package_share_directory(package_name), 'config')
    default_params_file = os.path.join(config_dir, 'default_params.yaml')
    
    # Launch arguments
    params_file = LaunchConfiguration('params_file')
    use_topic_sync = LaunchConfiguration('use_topic_sync')
    use_multi_threading = LaunchConfiguration('use_multi_threading')
    
    # Declare arguments
    arg_declarations = [
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to parameters file'
        ),
        DeclareLaunchArgument(
            'use_topic_sync',
            default_value='False',
            description='Whether to use topic synchronization'
        ),
        DeclareLaunchArgument(
            'use_multi_threading',
            default_value='True',
            description='Whether to use multi-threading'
        ),
    ]
    
    # Create tracking node - using different method
    tracking_node = Node(
        name='depth_tracking_node',
        namespace='',
        package='yolo_bench_d455',
        # Note: Referencing the Python module directly
        node_executable='tracking_node',  
        parameters=[{
            'use_topic_sync': use_topic_sync,
            'use_multi_threading': use_multi_threading,
        }],
        output='screen'
    )
    
    # Create FPS monitoring nodes
    # camera_fps_node = Node(
    #     name='camera_fps_monitor',
    #     namespace='',
    #     package='yolo_bench_d455',
    #     node_executable='performance_monitor',
    #     parameters=[{
    #         'topic': '/camera/camera/color/image_raw',
    #     }],
    #     output='screen'
    # )
    
    # depth_fps_node = Node(
    #     name='depth_fps_monitor',
    #     namespace='',
    #     package='yolo_bench_d455',
    #     node_executable='performance_monitor',
    #     parameters=[{
    #         'topic': '/camera/camera/depth/image_rect_raw',
    #     }],
    #     output='screen'
    # )

    return LaunchDescription(arg_declarations + [
        tracking_node
        # camera_fps_node,
        # depth_fps_node,
    ])