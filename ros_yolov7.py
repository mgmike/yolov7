#!/home/mike/anaconda3/envs/waymo/bin/python3

import rospy
import os
import sys
import json
from easydict import EasyDict as edict
import cv2

from yolov7 import Yolov7


dir_yolov7 = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_yolov7)

def main():
    configs = edict()
    # Add yolo configs
    with open(os.path.join(dir_yolov7, 'yolov7.json')) as j_object:
        configs.yolov7 = json.load(j_object)

    rospy.loginfo("Init yolov7")
    
    yolo = Yolov7(configs=configs.yolov7)

    rospy.init_node("yolov7", anonymous=True)

    while (False):
        if yolo.vis_img_rdy:
            rospy.loginfo("Got image")
            
            yolo.vis_img_rdy = False
            cv2.imshow('detections', yolo.vis_img)
            cv2.waitKey(0)  # 1 millisecondF
    rospy.spin()

        
if __name__ == '__main__':
    main()