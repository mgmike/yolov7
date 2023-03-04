#!/home/mike/anaconda3/envs/waymo/bin/python3

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from cv_bridge import CvBridge, CvBridgeError

import os
import sys
yolo_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(yolo_path)
sys.path.insert(0, '/home/mike/documents/cv_bridge_ws/install/lib')
print(sys.path)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import rospy
import numpy as np
from sensor_msgs.msg import Image

from threading import Thread

class Yolov7:
    def __init__(self, configs):
        self.configs = configs
        self.vis = True
        self.configs.trace = not self.configs.no_trace
        self.bridge = CvBridge()


        # Initialize
        set_logging()
        self.device = select_device(device=self.configs.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(os.path.join(yolo_path, self.configs.weights), map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.configs.img_size, s=stride)  # check img_size

        if self.configs.trace:
            self.model = TracedModel(self.model, self.device, self.configs.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        rospy.Subscriber("/carla/ego_vehicle/camera/rgb/front/image_color", Image, self.detect)

        rospy.loginfo("All set up")

        self.vis_img_rdy = False

    def detect(self, image):
        rospy.loginfo(f'Got an image of type {image.encoding}')
        try:
            im0 = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            im00 = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            img = self.bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
            rospy.loginfo(f'Img type: {type(img)}')
            # Play arond with interpolation types
            img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
            img = img.transpose(2, 0, 1)
            rospy.loginfo(f'Img type: {type(img)}')
            rospy.loginfo(f'Initial shape: ({image.height}, {image.width}) Converted shape: {img.shape}')
            img = np.ascontiguousarray(img)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
    
        # The default image
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.configs.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.configs.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.configs.conf_thres, self.configs.iou_thres, classes=self.configs.classes, agnostic=self.configs.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0)

        # Process detections
        for det in pred:  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.vis:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        self.vis_img = im00
        self.vis_img_rdy = True
        # Stream results
        # if self.vis:
        #     try:
        #         # cv_image_array = np.array(im0  , dtype = np.dtype('f8'))
        #         # cv_image_norm = np.array(im0  , dtype = np.dtype('f8'))
        #         # cv_image_norm = cv2.normalize(cv_image_norm, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        #         cv2.imshow('detections', im0)
        #         cv2.waitKey(25)  # 1 millisecondF
        #     finally:
        #         rospy.logerror('Issue with imshow')

    def get_boolean(self, str):
        return True if str == 'true' else False