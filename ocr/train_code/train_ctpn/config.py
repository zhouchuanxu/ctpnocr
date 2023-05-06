#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:09
#
# @Author: Greg Gao(laygin)
#'''
import os

# base_dir = 'path to dataset base dir'
base_dir = './images'
img_dir = os.path.join(base_dir, 'VOC2007_text_detection/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007_text_detection/Annotations')

icdar17_mlt_img_dir = './train_data/train_img/'
icdar17_mlt_gt_dir = './train_data/train_label/'
num_workers = 0


#pretrained_weights = 'D:/allproject/bishe/pytorch_tutorial-master/pytorch_tutorial-master/pytorch_tutorial/081_083/ocr/checkpoints/v3_ctpn_ep01_0.6968_0.2112_0.9080.pth'
pretrained_weights = 'D:/allproject/bishe/pytorch_tutorial-master/pytorch_tutorial-master/pytorch_tutorial/081_083/ocr/checkpoints/v3_ctpn_ep12_0.5663_0.1344_0.7006.pth'



anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

checkpoints_dir = 'D:/allproject/bishe/pytorch_tutorial-master/pytorch_tutorial-master/pytorch_tutorial/081_083/ocr/checkpoints'
#checkpoints_dir = './checkpoints'
outputs = r'./logs'
