# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import transforms3d
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import json




class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_box = self.create_publisher(String, 'box_xyz', 10)
        self.publisher_cone = self.create_publisher(String, 'cone_xyz', 10)
        self.camera_model()


    def depth_roi_callback(self, heading_angle, x:list,y:list,z:list,class_name):
        pub_msg = String()
        msg = dict()
        if class_name == "cone":
            msg['heading_angle'] = heading_angle
            msg['x_list'] = x
            msg['y_list'] = y
            msg['z_list'] = z
            msg['class_list'] = [0 for i in range(len(x))]
            pub_msg.data = json.dumps(msg)
            self.publisher_cone.publish(pub_msg)
            print("msg published",json.dumps(msg))
        elif class_name == "obstacle":
            msg['heading_angle'] = heading_angle
            msg['x_list'] = x
            msg['y_list'] = y
            msg['z_list'] = z
            msg['class_list'] = [1 for i in range(len(x))]
            pub_msg.data = json.dumps(msg)
            self.publisher_box.publish(pub_msg)
            print("msg published",json.dumps(msg))
    def camera_model(self):
        '''
        Spatial Tiny-yolo example
        Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
        Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
        '''

        # Get argument first
        nnBlobPath = str((Path(__file__).parent / Path('../models/box_cone_v9/Custom_Object_Detection_using_YOLOv5_512_DataSet_2_openvino_2022.1_10shave.blob')).resolve().absolute())
        # if 1 < len(sys.argv):
        #     arg = sys.argv[1]
        #     if arg == "yolo3":
        #         nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
        #     elif arg == "yolo4":
        #         nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
        #     else:
        #         nnBlobPath = arg
        # else:
        #     print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

        if not Path(nnBlobPath).exists():
            import sys
            raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

        # Tiny yolo v3/4 label texts
        labelMap = [
                    "cone",
                    "obstacle"
                ]

        syncNN = True

        lrcheck = True
        subpixel = True
        long_range = False

        # Create pipeline
        pipeline = dai.Pipeline()

        #IMU
        imu = pipeline.create(dai.node.IMU)
        xlinkOut = pipeline.create(dai.node.XLinkOut)
        xlinkOut.setStreamName("imu")
        # enable ROTATION_VECTOR at 400 hz rate
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 100)
        # it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
        # above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
        imu.setBatchReportThreshold(1)
        # maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
        # if lower or equal to batchReportThreshold then the sending is always blocking on device
        # useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
        imu.setMaxBatchReports(10)
        # Link plugins IMU -> XLINK
        imu.out.link(xlinkOut.input)

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
        camRgb.setPreviewSize(512, 320)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
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
        stereo.setLeftRightCheck(lrcheck)
        stereo.setSubpixel(subpixel)
        stereo.setExtendedDisparity(long_range)
        stereo.enableDistortionCorrection(True)
        stereo.setInputResolution(512,320)

        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.8)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.2)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(8000)

        spatialDetectionNetwork.setSpatialCalculationAlgorithm(dai.SpatialLocationCalculatorAlgorithm.MEDIAN)


        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(2)
        spatialDetectionNetwork.setCoordinateSize(4)
        spatialDetectionNetwork.setAnchors([
                        10.0,
                        13.0,
                        16.0,
                        30.0,
                        33.0,
                        23.0,
                        30.0,
                        61.0,
                        62.0,
                        45.0,
                        59.0,
                        119.0,
                        116.0,
                        90.0,
                        156.0,
                        198.0,
                        373.0,
                        326.0
                    ])
        spatialDetectionNetwork.setAnchorMasks({
                "side64": [
                    0,
                    1,
                    2
                ],
                "side32": [
                    3,
                    4,
                    5
                ],
                "side16": [
                    6,
                    7,
                    8
                ]
            })
        spatialDetectionNetwork.setIouThreshold(0.5)
        spatialDetectionNetwork.setConfidenceThreshold(0.2)

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

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
            baseTs = None
            base_yaw = None
            del_yaw = 0

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)
            printOutputLayersOnce = True

            while True:
                inPreview = previewQueue.get()
                inDet = detectionNNQueue.get()
                depth = depthQueue.get()
                inNN = networkQueue.get()

                imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived
                imuPackets = imuData.packets
                for imuPacket in imuPackets:
                    rVvalues = imuPacket.rotationVector
                    # rvTs = rVvalues.getTimestampDevice()
                    # if baseTs is None:
                    #     baseTs = rvTs
                    # rvTs = rvTs - baseTs
                    # imuF = "{:.06f}"
                    # tsF  = "{:.03f}"
                    yaw = transforms3d.euler.quat2euler([rVvalues.real,rVvalues.i,rVvalues.j,rVvalues.k])[2]
                    if base_yaw is None:
                        base_yaw = yaw
                    del_yaw = yaw-base_yaw
                    del_yaw = np.arctan2(np.sin(del_yaw),np.cos(del_yaw))

                if printOutputLayersOnce:
                    toPrint = 'Output layer names:'
                    for ten in inNN.getAllLayerNames():
                        toPrint = f'{toPrint} {ten},'
                    print(toPrint)
                    printOutputLayersOnce = False;

                frame = inPreview.getCvFrame()
                depthFrame = depth.getFrame() # depthFrame values are in millimeters

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                detections = inDet.detections

                # If the frame is available, draw bounding boxes on it and show the frame
                height = frame.shape[0]
                width  = frame.shape[1]
                cone_x = []
                cone_y = []
                cone_z = []
                box_x = []
                box_y = []
                box_z = []
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
                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                    # Denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)
                    try:
                        label = labelMap[detection.label]
                    except:
                        label = detection.label
                    if label == "cone":
                        cone_x.append(detection.spatialCoordinates.x)
                        cone_y.append(detection.spatialCoordinates.y)
                        cone_z.append(detection.spatialCoordinates.z)
                    if label == "obstacle":
                        box_x.append(detection.spatialCoordinates.x)
                        box_y.append(detection.spatialCoordinates.y)
                        box_z.append(detection.spatialCoordinates.z)

                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                if len(cone_x) != 0:
                    self.depth_roi_callback(del_yaw, cone_x,cone_y,cone_z,"cone")
                if len(box_x) != 0:
                    self.depth_roi_callback(del_yaw, box_x,box_y,box_z,"obstacle")

                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("rgb", frame)

                if cv2.waitKey(1) == ord('q'):
                    break


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
