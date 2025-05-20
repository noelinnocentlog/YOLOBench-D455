# File: yolo.launch.py
#!/usr/bin/env python3

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():

    def run_yolo(context, use_tracking, use_3d):

        use_tracking = eval(context.perform_substitution(use_tracking))
        use_3d = eval(context.perform_substitution(use_3d))

        # YOLO Model Parameters
        model_version = LaunchConfiguration("model_version")
        model_version_cmd = DeclareLaunchArgument(
            "model_version",
            default_value="yolov8",
            description="YOLO version to use: yolov5, yolov6, yolov7, yolov8",
        )

        model = LaunchConfiguration("model")
        model_cmd = DeclareLaunchArgument(
            "model",
            default_value="yolov8m.pt",
            description="Model name or path",
        )

        tracker = LaunchConfiguration("tracker")
        tracker_cmd = DeclareLaunchArgument(
            "tracker",
            default_value="bytetrack.yaml",
            description="Tracker name or path",
        )

        # Hardware/Performance Parameters
        device = LaunchConfiguration("device")
        device_cmd = DeclareLaunchArgument(
            "device",
            default_value="cuda:0",
            description="Device to use (GPU/CPU)",
        )

        half = LaunchConfiguration("half")
        half_cmd = DeclareLaunchArgument(
            "half",
            default_value="True",
            description="Use half-precision (FP16) inference for better performance on Jetson",
        )

        # Image Parameters
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

        threshold = LaunchConfiguration("threshold")
        threshold_cmd = DeclareLaunchArgument(
            "threshold",
            default_value="0.5",
            description="Minimum probability of a detection to be published",
        )

        iou = LaunchConfiguration("iou")
        iou_cmd = DeclareLaunchArgument(
            "iou",
            default_value="0.7",
            description="IoU threshold",
        )

        imgsz_height = LaunchConfiguration("imgsz_height")
        imgsz_height_cmd = DeclareLaunchArgument(
            "imgsz_height",
            default_value="480",  # Reduced from 640 for better Jetson performance
            description="Image height for inference",
        )

        imgsz_width = LaunchConfiguration("imgsz_width")
        imgsz_width_cmd = DeclareLaunchArgument(
            "imgsz_width",
            default_value="640",
            description="Image width for inference",
        )

        max_det = LaunchConfiguration("max_det")
        max_det_cmd = DeclareLaunchArgument(
            "max_det",
            default_value="100",  # Reduced from 300 for Jetson performance
            description="Maximum number of detections allowed per image",
        )

        augment = LaunchConfiguration("augment")
        augment_cmd = DeclareLaunchArgument(
            "augment",
            default_value="False",
            description="Whether to enable test-time augmentation (TTA) for predictions",
        )

        agnostic_nms = LaunchConfiguration("agnostic_nms")
        agnostic_nms_cmd = DeclareLaunchArgument(
            "agnostic_nms",
            default_value="False",
            description="Whether to enable class-agnostic Non-Maximum Suppression (NMS)",
        )

        retina_masks = LaunchConfiguration("retina_masks")
        retina_masks_cmd = DeclareLaunchArgument(
            "retina_masks",
            default_value="False",
            description="Whether to use high-resolution segmentation masks if available",
        )

        input_image_topic = LaunchConfiguration("input_image_topic")
        input_image_topic_cmd = DeclareLaunchArgument(
            "input_image_topic",
            default_value="/camera/color/image_raw",
            description="Name of the input image topic",
        )

        image_reliability = LaunchConfiguration("image_reliability")
        image_reliability_cmd = DeclareLaunchArgument(
            "image_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="QoS reliability for image topic (0=default, 1=Reliable, 2=Best Effort)",
        )

        input_depth_topic = LaunchConfiguration("input_depth_topic")
        input_depth_topic_cmd = DeclareLaunchArgument(
            "input_depth_topic",
            default_value="/camera/depth/image_rect_raw",
            description="Name of the input depth topic",
        )

        depth_image_reliability = LaunchConfiguration("depth_image_reliability")
        depth_image_reliability_cmd = DeclareLaunchArgument(
            "depth_image_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="QoS reliability for depth image topic (0=default, 1=Reliable, 2=Best Effort)",
        )

        input_depth_info_topic = LaunchConfiguration("input_depth_info_topic")
        input_depth_info_topic_cmd = DeclareLaunchArgument(
            "input_depth_info_topic",
            default_value="/camera/depth/camera_info",
            description="Name of the input depth info topic",
        )

        namespace = LaunchConfiguration("namespace")
        namespace_cmd = DeclareLaunchArgument(
            "namespace",
            default_value="yolo",
            description="Namespace for the nodes",
        )

        use_debug = LaunchConfiguration("use_debug")
        use_debug_cmd = DeclareLaunchArgument(
            "use_debug",
            default_value="True",
            description="Whether to activate the debug node",
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

        yolo_node_cmd = Node(
            package="yolo_ros",
            executable="yolo_node",
            name="yolo_node",
            namespace=namespace,
            parameters=[
                {
                    "model_version": model_version,
                    "model": model,
                    "device": device,
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
                }
            ],
            remappings=[("image_raw", input_image_topic)],
        )

        tracking_node_cmd = Node(
            package="yolo_ros",
            executable="tracking_node",
            name="tracking_node",
            namespace=namespace,
            parameters=[{"tracker": tracker, "image_reliability": image_reliability}],
            remappings=[("image_raw", input_image_topic)],
            condition=IfCondition(PythonExpression([str(use_tracking)])),
        )

        detect_3d_node_cmd = Node(
            package="yolo_ros",
            executable="detect_3d_node",
            name="detect_3d_node",
            namespace=namespace,
            parameters=[
                {
                    "depth_image_units_divisor": 1000,
                    "depth_image_reliability": depth_image_reliability,
                    "depth_info_reliability": depth_image_reliability,
                }
            ],
            remappings=[
                ("depth_image", input_depth_topic),
                ("depth_info", input_depth_info_topic),
                ("detections", detect_3d_detections_topic),
            ],
            condition=IfCondition(PythonExpression([str(use_3d)])),
        )

        debug_node_cmd = Node(
            package="yolo_ros",
            executable="debug_node",
            name="debug_node",
            namespace=namespace,
            parameters=[{"image_reliability": image_reliability}],
            remappings=[
                ("image_raw", input_image_topic),
                ("detections", debug_detections_topic),
            ],
            condition=IfCondition(PythonExpression([use_debug])),
        )

        return (
            model_version_cmd,
            model_cmd,
            tracker_cmd,
            device_cmd,
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
            namespace_cmd,
            use_debug_cmd,
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