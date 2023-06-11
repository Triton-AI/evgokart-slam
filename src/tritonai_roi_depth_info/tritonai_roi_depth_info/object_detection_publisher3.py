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

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import torch
import cv2
import depthai as dai
import numpy as np
import transforms3d

from .calc import HostSpatialsCalc
from .utility import *

import json
from pathlib import Path
import math


class ObjectDetectionPublisher(Node):

    def __init__(self):
        super().__init__('object_detection_publisher')
        self.publisher_box = self.create_publisher(String, 'box_xyz', 10)
        self.publisher_cone = self.create_publisher(String, 'cone_xyz', 10)
        self.camRes = dai.ColorCameraProperties.SensorResolution.THE_800_P
        self.camSocket = dai.CameraBoardSocket.RGB
        self.ispScale = (1,2)
        self.camera_model()


    def getMesh(self, calibData, ispSize):
        M1 = np.array(calibData.getCameraIntrinsics(self.camSocket, ispSize[0], ispSize[1]))
        d1 = np.array(calibData.getDistortionCoefficients(self.camSocket))
        print(d1)

        # M1 = np.array([
        #     [576.7072143554688, 0.0, 632.15673828125],
        #     [0.0, 576.8490600585938, 401.6389465332031],
        #     [0.0, 0.0, 1.0]
        # ])

        R1 = np.identity(3)
        mapX, mapY = cv2.initUndistortRectifyMap(M1, d1, R1, M1, ispSize, cv2.CV_32FC1)

        meshCellSize = 16
        mesh0 = []
        # Creates subsampled mesh which will be loaded on to device to undistort the image
        for y in range(mapX.shape[0] + 1): # iterating over height of the image
            if y % meshCellSize == 0:
                rowLeft = []
                for x in range(mapX.shape[1]): # iterating over width of the image
                    if x % meshCellSize == 0:
                        if y == mapX.shape[0] and x == mapX.shape[1]:
                            rowLeft.append(mapX[y - 1, x - 1])
                            rowLeft.append(mapY[y - 1, x - 1])
                        elif y == mapX.shape[0]:
                            rowLeft.append(mapX[y - 1, x])
                            rowLeft.append(mapY[y - 1, x])
                        elif x == mapX.shape[1]:
                            rowLeft.append(mapX[y, x - 1])
                            rowLeft.append(mapY[y, x - 1])
                        else:
                            rowLeft.append(mapX[y, x])
                            rowLeft.append(mapY[y, x])
                if (mapX.shape[1] % meshCellSize) % 2 != 0:
                    rowLeft.append(0)
                    rowLeft.append(0)

                mesh0.append(rowLeft)

        mesh0 = np.array(mesh0)
        meshWidth = mesh0.shape[1] // 2
        meshHeight = mesh0.shape[0]
        mesh0.resize(meshWidth * meshHeight, 2)
        mesh = list(map(tuple, mesh0))

        return mesh, meshWidth, meshHeight


    def create_pipeline(self, calibData):
        # Create pipeline
        pipeline = dai.Pipeline()


        cam = pipeline.create(dai.node.ColorCamera)
        cam.setIspScale(self.ispScale)
        cam.setBoardSocket(self.camSocket)
        cam.setResolution(self.camRes)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        manip = pipeline.create(dai.node.ImageManip)
        mesh, meshWidth, meshHeight = self.getMesh(calibData, cam.getIspSize())
        manip.setWarpMesh(mesh, meshWidth, meshHeight)
        manip.setMaxOutputFrameSize(cam.getIspWidth() * cam.getIspHeight() * 3 // 2)
        cam.isp.link(manip.inputImage)

        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName("rgb")
        manip.out.link(cam_xout.input)

        # dist_xout = pipeline.create(dai.node.XLinkOut)
        # dist_xout.setStreamName("Distorted")
        # cam.isp.link(dist_xout.input)
        # cam.ispsetPreviewSize(256, 160)

        return pipeline



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
        Performs inference on RGB camera frame and calculates spatial location coordinates: x,y,z relative to the center of depth map.
        '''
        
        with dai.Device() as device:
            model = torch.hub.load('/home/projects/ultralytics_yolov5_master', 'custom', path=str((Path(__file__).parent / Path('../models/box_cone_v9/Custom_Object_Detection_using_YOLOv5_512_DataSet_2.pt')).resolve().absolute()), source='local', force_reload=True)
            model.conf = 0.5  # NMS confidence threshold
            model.iou = 0.5  # NMS IoU threshold
            # model.agnostic = False  # NMS class-agnostic
            # model.multi_label = False  # NMS multiple labels per box
            # model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
            model.max_det = 20  # maximum number of detections per image
            # model.amp = False  # Automatic Mixed Precision (AMP) inference
            color = (255, 255, 255)

            # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
            extended_disparity = True
            # Better accuracy for longer distance, fractional disparity 32-levels:
            subpixel = True
            # Better handling for occlusions:
            lr_check = True

            calibData = device.readCalibration()
            pipeline = self.create_pipeline(calibData)


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

            # Define sources and outputs
            monoLeft = pipeline.create(dai.node.MonoCamera)
            monoRight = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)
            spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

            xoutDepth = pipeline.create(dai.node.XLinkOut)
            xoutSpatialData = pipeline.create(dai.node.XLinkOut)
            xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

            xoutDepth.setStreamName("depth")
            xoutSpatialData.setStreamName("spatialData")
            xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

            # Properties
            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
            monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(False)
            stereo.setSubpixel(True)
            spatialLocationCalculator.inputConfig.setWaitForMessage(False)


            #Config
            topLeft = dai.Point2f(0.4, 0.4)
            bottomRight = dai.Point2f(0.6, 0.6)
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 100
            config.depthThresholds.upperThreshold = 10000
            calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN

            config.roi = dai.Rect(topLeft, bottomRight)

            spatialLocationCalculator.inputConfig.setWaitForMessage(False)
            spatialLocationCalculator.initialConfig.addROI(config)

            # Linking
            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)

            spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
            stereo.depth.link(spatialLocationCalculator.inputDepth)

            spatialLocationCalculator.out.link(xoutSpatialData.input)
            xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)


            device.startPipeline(pipeline)

            text = TextHelper()
            hostSpatials = HostSpatialsCalc(device)
            
            print('Connected cameras:', device.getConnectedCameraFeatures())
            # Print out usb speed
            print('Usb speed:', device.getUsbSpeed().name)
            # Bootloader version
            if device.getBootloaderVersion() is not None:
                print('Bootloader version:', device.getBootloaderVersion())
            # Device name
            print('Device name:', device.getDeviceName())

            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
            spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
            color = (0,200,40)
            fontType = cv2.FONT_HERSHEY_TRIPLEX

            base_yaw = None
            del_yaw = 0

            diffs = np.array([])
            
            while True:

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

                inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
                
                # Inference
                frame = inRgb.getCvFrame()
                results = model(frame[:, :, ::-1])  # includes NMS
                
                depthData = depthQueue.get()
                depthFrame = depthData.getFrame()

                # cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
                # cvColorMap[0] = [0, 0, 0]
                # maxDisp = stereo.initialConfig.getMaxDisparity()
                # depthFrameColor = (depthFrame * (255.0 / maxDisp)).astype(np.uint8)
                # depthFrameColor = cv2.applyColorMap(depthFrameColor, cvColorMap)

                # depth_downscaled = depthFrame[::4]
                # min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
                # max_depth = np.percentile(depth_downscaled, 99)
                depthFrameColor = np.interp(depthFrame, (100, 15000), (0, 255)).astype(np.uint8)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                cone_x = []
                cone_y = []
                cone_z = []
                box_x = []
                box_y = []
                box_z = []
                x1,y1,x2,y2 = (0,0,0,0)
                x = 0
                y = 0
                spa_x = float('nan')
                spa_y = float('nan')
                spa_z = float('nan')
                cfg = dai.SpatialLocationCalculatorConfig()
                for xyxy in results.xyxy[0]:
                    # print(xyxy)
                    try:
                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        config.roi =  dai.Rect(dai.Point2f(x1,y1),dai.Point2f(x2,y2))
                        cfg.addROI(config)
                        spatialCalcConfigInQueue.send(cfg)
                        conf = float(xyxy[4])*100
                        label = ['cone','obstacle'][int(xyxy[5])]
                        spatials, centroid = hostSpatials.calc_spatials(depthData, (x1,y1,x2,y2)) # roi --> mean spacial & centroid
                        x = centroid["x"]
                        y = centroid["y"]
                        spa_x = np.round(spatials['x'], 2)
                        spa_y = np.round(spatials['y'], 2)
                        spa_z = np.round(spatials['z'], 2)
                        if label == "cone":
                            cone_x.append(spa_x)
                            cone_y.append(spa_y)
                            cone_z.append(spa_z)
                        if label == "obstacle":
                            box_x.append(spa_x)
                            box_y.append(spa_y)
                            box_z.append(spa_z)
                    except Exception as e:
                        print(e)
                        continue
                
                spatialData = spatialCalcQueue.get().getSpatialLocations()
                for depthData in spatialData:
                    roi = depthData.config.roi
                    roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    depthMin = depthData.depthMin
                    depthMax = depthData.depthMax

                    fontType = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.rectangle(depthFrameColor, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (x1 + 10, y1 + 20), fontType, 0.5, color)
                    cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (x1 + 10, y1 + 35), fontType, 0.5, color)
                    cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (x1 + 10, y1 + 50), fontType, 0.5, color)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(frame, f"X: {int(depthData.spatialCoordinates.x)} mm", (x1 + 10, y1 + 20), fontType, 0.5, color)
                    cv2.putText(frame, f"Y: {int(depthData.spatialCoordinates.y)} mm", (x1 + 10, y1 + 35), fontType, 0.5, color)
                    cv2.putText(frame, f"Z: {int(depthData.spatialCoordinates.z)} mm", (x1 + 10, y1 + 50), fontType, 0.5, color)

                # Show the frame
                cv2.imshow("depth", depthFrameColor)
                cv2.imshow("tracker", frame)

                latencyMs = (dai.Clock.now() - inRgb.getTimestamp()).total_seconds() * 1000
                diffs = np.append(diffs, latencyMs)
                # print('Latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}'.format(latencyMs, np.average(diffs), np.std(diffs)))

                if len(cone_x) != 0:
                    self.depth_roi_callback(del_yaw, cone_x,cone_y,cone_z,"cone")
                if len(box_x) != 0:
                    self.depth_roi_callback(del_yaw, box_x,box_y,box_z,"obstacle")

                if cv2.waitKey(1) == ord('q'):
                    break



def main(args=None):
    rclpy.init(args=args)

    obj_det_publisher = ObjectDetectionPublisher()
    rclpy.spin(obj_det_publisher)

    obj_det_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
