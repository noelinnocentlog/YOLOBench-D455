# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modified for Jetson AGX Xavier compatibility

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():

    def run_yolo(context, use_tracking, use_3d):

        use_tracking = eval(context.perform_substitution(use_tracking))
        use_3d = eval(context.perform_substitution(use_3d))

        # Model parameters - optimized for Xavier
        model_type = LaunchConfiguration("model_type")
        model_type_cmd = DeclareLaunchArgument(
            "model_type",
            default_value="YOLO",
            description="Model type form Ultralytics (YOLO, World)",
        )

        model = LaunchConfiguration("model")
        model_cmd = DeclareLaunchArgument(
            "model",
            default_value="yolov8n.pt",  # Small model by default for Xavier
            description="Model name or path",
        )

        tracker = LaunchConfiguration("tracker")
        tracker_cmd = DeclareLaunchArgument(
            "tracker",
            default_value="bytetrack.yaml",
            description="Tracker name or path",
        )

        device = LaunchConfiguration("device")
        device_cmd = DeclareLaunchArgument(
            "device",
            default_value="cuda:0",
            description="Device to use (GPU/CPU)",
        )

        # Xavier performance optimizations
        use_tensorrt = LaunchConfiguration("use_tensorrt")
        use_tensorrt_cmd = DeclareLaunchArgument(
            "use_tensorrt",
            default_value="True",
            description="Use TensorRT for acceleration",
        )

        tensorrt_fp16 = LaunchConfiguration("tensorrt_fp16")
        tensorrt_fp16_cmd = DeclareLaunchArgument(
            "tensorrt_fp16",
            default_value="True",
            description="Use FP16 precision for TensorRT",
        )

        process_every_n_frame = LaunchConfiguration("process_every_n_frame")
        process_every_n_frame_cmd = DeclareLaunchArgument(
            "process_every_n_frame",
            default_value="1",
            description="Process every Nth frame (for performance)",
        )

        yolo_encoding = LaunchConfiguration("yolo_encoding")
        yolo_encoding_cmd = DeclareLaunchArgument(
            "yolo_encoding",
            default_value="bgr8",
            description="Encoding of the input image topic",
        )

        enable = LaunchConfiguration("enable")
        enable_cmd = DeclareLaunchArgument(
            "enable",
            default_value="True",
            description="Whether to start YOLO enabled",
        )

        # Lower threshold for real-time performance
        threshold = LaunchConfiguration("threshold")
        threshold_cmd = DeclareLaunchArgument(
            "threshold",
            default_value="0.45",
            description="Minimum probability of a detection to be published",
        )

        iou = LaunchConfiguration("iou")
        iou_cmd = DeclareLaunchArgument(
            "iou",
            default_value="0.45",
            description="IoU threshold",
        )

        # Lower image size for better performance on Xavier
        imgsz_height = LaunchConfiguration("imgsz_height")
        imgsz_height_cmd = DeclareLaunchArgument(
            "imgsz_height",
            default_value="384",
            description="Image height for inference",
        )

        imgsz_width = LaunchConfiguration("imgsz_width")
        imgsz_width_cmd = DeclareLaunchArgument(
            "imgsz_width",
            default_value="384",
            description="Image width for inference",
        )

        # Always use half precision on Xavier
        half = LaunchConfiguration("half")
        half_cmd = DeclareLaunchArgument(
            "half",
            default_value="True",
            description="Whether to enable half-precision (FP16) inference speeding up model inference",
        )

        # Lower max detections for better performance
        max_det = LaunchConfiguration("max_det")
        max_det_cmd = DeclareLaunchArgument(
            "max_det",
            default_value="100",
            description="Maximum number of detections allowed per image",
        )

        # Disable augmentation for performance
        augment = LaunchConfiguration("augment")
        augment_cmd = DeclareLaunchArgument(
            "augment",
            default_value="False",
            description="Whether to enable test-time augmentation",
        )

        agnostic_nms = LaunchConfiguration("agnostic_nms")
        agnostic_nms_cmd = DeclareLaunchArgument(
            "agnostic_nms",
            default_value="False",
            description="Whether to enable class-agnostic NMS",
        )

        # Disable retina masks for performance
        retina_masks = LaunchConfiguration("retina_masks")
        retina_masks_cmd = DeclareLaunchArgument(
            "retina_masks",
            default_value="False",
            description="Whether to use high-resolution segmentation masks",
        )

        input_image_topic = LaunchConfiguration("input_image_topic")
        input_image_topic_cmd = DeclareLaunchArgument(
            "input_image_topic",
            default_value="/camera/camera/color/image_raw",
            description="Name of the input image topic",
        )

        image_reliability = LaunchConfiguration("image_reliability")
        image_reliability_cmd = DeclareLaunchArgument(
            "image_reliability",
            default_value="2",  # Best effort for Xavier
            choices=["0", "1", "2"],
            description="Reliability QoS (0=default, 1=Reliable, 2=Best Effort)",
        )

        input_depth_topic = LaunchConfiguration("input_depth_topic")
        input_depth_topic_cmd = DeclareLaunchArgument(
            "input_depth_topic",
            default_value="/camera/camera/depth/image_rect_raw",
            description="Name of the input depth topic",
        )

        depth_image_reliability = LaunchConfiguration("depth_image_reliability")
        depth_image_reliability_cmd = DeclareLaunchArgument(
            "depth_image_reliability",
            default_value="2",  # Best effort for Xavier
            choices=["0", "1", "2"],
            description="Reliability QoS for depth image",
        )

        input_depth_info_topic = LaunchConfiguration("input_depth_info_topic")
        input_depth_info_topic_cmd = DeclareLaunchArgument(
            "input_depth_info_topic",
            default_value="/camera/camera/depth/camera_info",
            description="Name of the input depth info topic",
        )

        depth_info_reliability = LaunchConfiguration("depth_info_reliability")
        depth_info_reliability_cmd = DeclareLaunchArgument(
            "depth_info_reliability",
            default_value="2",  # Best effort for Xavier
            choices=["0", "1", "2"],
            description="Reliability QoS for depth info",
        )

        target_frame = LaunchConfiguration("target_frame")
        target_frame_cmd = DeclareLaunchArgument(
            "target_frame",
            default_value="base_link",
            description="Target frame to transform the 3D boxes",
        )

        depth_image_units_divisor = LaunchConfiguration("depth_image_units_divisor")
        depth_image_units_divisor_cmd = DeclareLaunchArgument(
            "depth_image_units_divisor",
            default_value="1000",
            description="Divisor used to convert raw depth to metres",
        )

        maximum_detection_threshold = LaunchConfiguration("maximum_detection_threshold")
        maximum_detection_threshold_cmd = DeclareLaunchArgument(
            "maximum_detection_threshold",
            default_value="0.3",
            description="Maximum detection threshold in the z axis",
        )

        namespace = LaunchConfiguration("namespace")
        namespace_cmd = DeclareLaunchArgument(
            "namespace",
            default_value="yolo",
            description="Namespace for the nodes",
        )
        
        # Xavier optimization: Simpler debug output
        use_debug = LaunchConfiguration("use_debug")
        use_debug_cmd = DeclareLaunchArgument(
            "use_debug",
            default_value="True",
            description="Whether to activate the debug node",
        )
        
        # Additional Xavier optimization parameters
        performance_overlay = LaunchConfiguration("performance_overlay")
        performance_overlay_cmd = DeclareLaunchArgument(
            "performance_overlay",
            default_value="True",
            description="Whether to show performance stats",
        )
        
        tracker_fps = LaunchConfiguration("tracker_fps")
        tracker_fps_cmd = DeclareLaunchArgument(
            "tracker_fps",
            default_value="15.0",
            description="Framerate for the tracker",
        )
        
        roi_downsample = LaunchConfiguration("roi_downsample")
        roi_downsample_cmd = DeclareLaunchArgument(
            "roi_downsample",
            default_value="2",
            description="Downsample factor for depth ROI",
        )

        # Get topics for remap
        detect_3d_detections_topic = "detections"
        debug_detections_topic = "detections"

        if use_tracking:
            detect_3d_detections_topic = "tracking"

        if use_tracking and not use_3d:
            debug_detections_topic = "tracking"
        elif use_3d:
            debug_detections_topic = "detections_3d"

        # YOLO Node with Xavier optimizations
        yolo_node_cmd = Node(
            package="yolo_ros",
            executable="yolo_node",
            name="yolo_node",
            namespace=namespace,
            parameters=[
                {
                    "model_type": model_type,
                    "model": model,
                    "device": device,
                    "use_tensorrt": use_tensorrt,
                    "tensorrt_fp16": tensorrt_fp16,
                    "yolo_encoding": yolo_encoding,
                    "enable": enable,
                    "threshold": threshold,
                    "iou": iou,
                    "imgsz_height": imgsz_height,
                    "imgsz_width": imgsz_width,
                    "half": half,
                    "max_det": max_det,
                    "augment": augment,
                    "agnostic_nms": agnostic_nms,
                    "retina_masks": retina_masks,
                    "image_reliability": image_reliability,
                    "performance_overlay": performance_overlay,
                }
            ],
            remappings=[("image_raw", input_image_topic)],
        )

        # Tracking Node with Xavier optimizations
        tracking_node_cmd = Node(
            package="yolo_ros",
            executable="tracking_node",
            name="tracking_node",
            namespace=namespace,
            parameters=[{
                "tracker": tracker,
                "image_reliability": image_reliability,
                "process_every_n_frame": process_every_n_frame,
                "tracker_fps": tracker_fps,
                "performance_monitor": performance_overlay,
            }],
            remappings=[("image_raw", input_image_topic)],
            condition=IfCondition(PythonExpression([str(use_tracking)])),
        )

        # 3D Detection Node with Xavier optimizations
        detect_3d_node_cmd = Node(
            package="yolo_ros",
            executable="detect_3d_node",
            name="detect_3d_node",
            namespace=namespace,
            parameters=[
                {
                    "target_frame": target_frame,
                    "maximum_detection_threshold": maximum_detection_threshold,
                    "depth_image_units_divisor": depth_image_units_divisor,
                    "depth_image_reliability": depth_image_reliability,
                    "depth_info_reliability": depth_info_reliability,
                    "process_every_n_frame": process_every_n_frame,
                    "roi_downsample": roi_downsample,
                    "debug_mode": False,
                }
            ],
            remappings=[
                ("depth_image", input_depth_topic),
                ("depth_info", input_depth_info_topic),
                ("detections", detect_3d_detections_topic),
            ],
            condition=IfCondition(PythonExpression([str(use_3d)])),
        )

        # Debug Node with Xavier optimizations
        debug_node_cmd = Node(
            package="yolo_ros",
            executable="debug_node",
            name="debug_node",
            namespace=namespace,
            parameters=[{
                "image_reliability": image_reliability,
                "enable_markers": True,
                "enable_keypoints": True,
                "enable_masks": True,
                "text_scale": 0.6, # Smaller text for performance
                "line_thickness": 1, # Thinner lines for performance
                "performance_overlay": performance_overlay,
            }],
            remappings=[
                ("image_raw", input_image_topic),
                ("detections", debug_detections_topic),
            ],
            condition=IfCondition(PythonExpression([use_debug])),
        )

        return (
            model_type_cmd,
            model_cmd,
            tracker_cmd,
            device_cmd,
            use_tensorrt_cmd,
            tensorrt_fp16_cmd,
            yolo_encoding_cmd,
            enable_cmd,
            threshold_cmd,
            iou_cmd,
            imgsz_height_cmd,
            imgsz_width_cmd,
            half_cmd,
            max_det_cmd,
            augment_cmd,
            agnostic_nms_cmd,
            retina_masks_cmd,
            input_image_topic_cmd,
            image_reliability_cmd,
            input_depth_topic_cmd,
            depth_image_reliability_cmd,
            input_depth_info_topic_cmd,
            depth_info_reliability_cmd,
            target_frame_cmd,
            depth_image_units_divisor_cmd,
            maximum_detection_threshold_cmd,
            namespace_cmd,
            use_debug_cmd,
            performance_overlay_cmd,
            tracker_fps_cmd,
            process_every_n_frame_cmd,
            roi_downsample_cmd,
            yolo_node_cmd,
            tracking_node_cmd,
            detect_3d_node_cmd,
            debug_node_cmd,
        )

    use_tracking = LaunchConfiguration("use_tracking")
    use_tracking_cmd = DeclareLaunchArgument(
        "use_tracking",
        default_value="True",
        description="Whether to activate tracking",
    )

    use_3d = LaunchConfiguration("use_3d")
    use_3d_cmd = DeclareLaunchArgument(
        "use_3d",
        default_value="False",
        description="Whether to activate 3D detections",
    )

    return LaunchDescription(
        [
            use_tracking_cmd,
            use_3d_cmd,
            OpaqueFunction(function=run_yolo, args=[use_tracking, use_3d]),
        ]
    )