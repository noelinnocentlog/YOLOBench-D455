#!/usr/bin/env python3

# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
# ROS 2 Foxy compatible imports
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle.node import LifecycleState  # Changed path for Foxy

import cv2
import numpy as np
import message_filters
from cv_bridge import CvBridge

# Import with error handling for compatibility
try:
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.basetrack import BaseTrack
    from ultralytics.trackers import BOTSORT, BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, yaml_load
    from ultralytics.utils.checks import check_requirements, check_yaml
except ImportError:
    print("Installing required packages...")
    import os
    os.system("pip install ultralytics lap")
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.basetrack import BaseTrack
    from ultralytics.trackers import BOTSORT, BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, yaml_load
    from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class TrackingNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("tracking_node")

        # params
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.cv_bridge = CvBridge()
        
        # Initialize attributes used in on_deactivate
        self.image_sub = None
        self.detections_sub = None
        self._synchronizer = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        tracker_name = self.get_parameter("tracker").get_parameter_value().string_value

        self.image_reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )

        self.tracker = self.create_tracker(tracker_name)
        self._pub = self.create_publisher(DetectionArray, "tracking", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        image_qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # subs
        self.image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.detections_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        # Check if subscribers exist before destroying them
        if self.image_sub and hasattr(self.image_sub, 'sub'):
            self.destroy_subscription(self.image_sub.sub)
            
        if self.detections_sub and hasattr(self.detections_sub, 'sub'):
            self.destroy_subscription(self.detections_sub.sub)

        if self._synchronizer:
            del self._synchronizer
            self._synchronizer = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        if hasattr(self, 'tracker'):
            del self.tracker

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:
        """Create object tracker from YAML config."""
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in [
            "bytetrack",
            "botsort",
        ], f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def detections_cb(self, img_msg: Image, detections_msg: DetectionArray) -> None:
        """Process detections and perform tracking."""
        tracked_detections_msg = DetectionArray()
        tracked_detections_msg.header = img_msg.header

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # parse detections
        detection_list = []
        detection: Detection
        for detection in detections_msg.detections:
            detection_list.append(
                [
                    detection.bbox.center.position.x - detection.bbox.size.x / 2,
                    detection.bbox.center.position.y - detection.bbox.size.y / 2,
                    detection.bbox.center.position.x + detection.bbox.size.x / 2,
                    detection.bbox.center.position.y + detection.bbox.size.y / 2,
                    detection.score,
                    detection.class_id,
                ]
            )

        # tracking
        if len(detection_list) > 0:
            try:
                det = Boxes(np.array(detection_list), (img_msg.height, img_msg.width))
                tracks = self.tracker.update(det, cv_image)

                if len(tracks) > 0:
                    for t in tracks:
                        # Get tracked box
                        tracked_box = Boxes(t[:-1], (img_msg.height, img_msg.width))
                        t_idx = int(t[-1])
                        
                        # Check if index is valid
                        if t_idx < len(detections_msg.detections):
                            tracked_detection: Detection = detections_msg.detections[t_idx]

                            # get boxes values
                            box = tracked_box.xywh[0]
                            tracked_detection.bbox.center.position.x = float(box[0])
                            tracked_detection.bbox.center.position.y = float(box[1])
                            tracked_detection.bbox.size.x = float(box[2])
                            tracked_detection.bbox.size.y = float(box[3])

                            # get track id
                            track_id = ""
                            if hasattr(tracked_box, 'is_track') and tracked_box.is_track:
                                track_id = str(int(tracked_box.id))
                            tracked_detection.id = track_id

                            # append msg
                            tracked_detections_msg.detections.append(tracked_detection)
            except Exception as e:
                self.get_logger().error(f"Error in tracking: {str(e)}")

        # publish detections
        self._pub.publish(tracked_detections_msg)


def main():
    rclpy.init()
    node = TrackingNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()