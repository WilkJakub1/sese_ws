#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import sys
import cv2
import depthai as dai
import numpy as np
import time
# import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import ObjectHypothesisWithPose
from pathlib import Path
from labels import labelMap

class DetectionSubscriber(Node):
    super().__init__()
    def __init__(self):
        detection_subscriber = self.create_subscription(ObjectHypothesisWithPose, topic='/asv_vision/front_cam/detections', self.callback, 10)

    def callback(msg):
        pass

def main(args=None):

    # depthFrameColor = cv2.normalize(
    #         depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
    #     )
    # depthFrameColor = cv2.equalizeHist(depthFrameColor)
    # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

    width = 640
    height = 640

    for detection in detections:
        roiData = detection.boundingBoxMapping
        roi = roiData.roi
        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        xmin = int(topLeft.x)
        ymin = int(topLeft.y)
        xmax = int(bottomRight.x)
        ymax = int(bottomRight.y)
        cv2.rectangle(
            depthFrameColor,
            (xmin, ymin),
            (xmax, ymax),
            color,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        )

        # Denormalize bounding box
        x1 = int(detection.xmin * width)
        x2 = int(detection.xmax * width)
        y1 = int(detection.ymin * height)
        y2 = int(detection.ymax * height)
        try:
            label = labelMap[detection.label]
        except:
            label = detection.label
        cv2.putText(
            frame,
            str(label),
            (x1 + 10, y1 + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.putText(
            frame,
            "{:.2f}".format(detection.confidence * 100),
            (x1 + 10, y1 + 35),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.putText(
            frame,
            f"X: {int(detection.spatialCoordinates.x)} mm",
            (x1 + 10, y1 + 50),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.putText(
            frame,
            f"Y: {int(detection.spatialCoordinates.y)} mm",
            (x1 + 10, y1 + 65),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.putText(
            frame,
            f"Z: {int(detection.spatialCoordinates.z)} mm",
            (x1 + 10, y1 + 80),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        #
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        # What we want from each detection is:
        # labelMap[detection.label]
        # spatialCoordinates are in milimeters
        # detection.spatialCoordinates.x / 100 - we divide by 100, because we want to to map object with 10cm resolution
        # detection.spatialCoordinates.z / 100 - we divide by 100, because we want to to map object with 10cm resolution
        # msgD = msgDetection(label, detection.spatialCoordinates.x / 100, detection.spatialCoordinates.z / 100)
        msgD = ObjectHypothesisWithPose()
        msgD.id = detection.label
        msgD.pose.pose.position.x = detection.spatialCoordinates.x / 100
        msgD.pose.pose.position.y = detection.spatialCoordinates.y / 100
        msgD.pose.pose.position.z = detection.spatialCoordinates.z / 100
        detection_publisher.object_publisher.publish(msgD)
        # object_pub(msgD, object_publisher)
        # TODO think about placement of a rate.sleep()
        rate.sleep()

    # frame_pub(bridge, frame, frame_publisher)
    detection_publisher.frame_publisher.publish(bridge.cv2_to_imgmsg(frame))
    detection_publisher
    rate.sleep()

    # cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
    cv2.putText(
        frame,
        "NN fps: {:.2f}".format(fps),
        (2, frame.shape[0] - 4),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.4,
        color,
    )
    # cv2.imshow("depth", depthFrameColor)
    cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord("q"):
        break

if __name__ == '__main__':
    main()