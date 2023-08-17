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

"""
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks, but currently we tests it for tiny-yolo-v7
"""

camera_id = ""

class DetectionPublisher(Node):
    super().__init__()
    def __init__(self):
        object_publisher = self.create_publisher( ObjectHypothesisWithPose, '/asv_vision/front_cam/detections', 10)
        frame_publisher = self.create_publisher(Image, '/asv_vision/front_cam/raw', 10)


def object_pub(msg, publisher):
    publisher.publish(msg)

def frame_pub(bridge, frame, publisher):
    try:
        msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        publisher.publish(msg)
    except CvBridgeError as e:
        print(e)

def main(args=None):
    rclpy.init(args=args)

    # Loading a blob file of our yolov7
    nnBlobPath = '../DepthAI_model/roboyolo_openvino_2021.4_6shave.blob'

    if not Path(nnBlobPath).exists():
        import sys

        raise FileNotFoundError(
            f'Required file/s not found, please run "{sys.executable} install_requirements.py"'
        )

    # Current set of labels texts for 9 classes
    # labelMap = ['buoy_red', 'buoy_green', 'buoy_blue', 'buoy_yellow', 'buoy_black', 'marker_red', 'marker_green', 'pillar', 'dock']

    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()
    pipeline.setXLinkChunkSize(64 * 1024)
    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    nnNetworkOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")
    nnNetworkOut.setStreamName("nnNetwork")

    # Properties
    camRgb.setPreviewSize(
        640, 640
    )  # Size of images (640, 640) is required by DepthAI model conversion pipeline
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(9)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    spatialDetectionNetwork.setAnchorMasks(
        {"side80": [1, 2, 3], "side40": [3, 4, 5], "side20": [3, 4, 5]}
    )
    spatialDetectionNetwork.setIouThreshold(0.5)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)
    if syncNN:
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
    spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

    detection_publisher = DetectionPublisher()

    # Set rospy node
    # rospy.init_node('CameraNode', anonymous=True)
    # Set rospy rate
    # rate = rospy.Rate(30) # 1hz
    # Connect to device and start pipeline - this is the main loop of a program
    device_info = dai.DeviceInfo(camera_id)

    with dai.Device(pipeline, device_info) as device:


        bridge = CvBridge()
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(
            name="detections", maxSize=4, blocking=False
        )
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)

        # It uses USB3
        # print(f'USB speed = {device.getUsbSpeed()}')
        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        printOutputLayersOnce = True

        #while True:
        while True:
            inPreview = previewQueue.get()
            inDet = detectionNNQueue.get()
            depth = depthQueue.get()
            inNN = networkQueue.get()

            if printOutputLayersOnce:
                toPrint = "Output layer names:"
                for ten in inNN.getAllLayerNames():
                    toPrint = f"{toPrint} {ten},"
                print(toPrint)
                printOutputLayersOnce = False

            frame = inPreview.getCvFrame()
            depthFrame = depth.getFrame()  # depthFrame values are in millimeters

            depthFrameColor = cv2.normalize(
                depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            counter += 1
            current_time = time.monotonic()
            if (current_time - startTime) > 1:
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            detections = inDet.detections

            # If the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width = frame.shape[1]
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

        rclpy.spin(detection_publisher)
        
    detection_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()