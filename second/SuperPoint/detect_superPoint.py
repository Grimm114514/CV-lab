"""
SuperPoint 特征点检测，填充特征匹配流程
"""

from superpoint import SuperPoint
import cv2
import numpy as np
import torch
from utils import *
import json
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": "superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        print("SuperPoint detector config: ")
        print(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        print("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # print("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": [image.shape[0], image.shape[1]],
            # "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }

        return ret_dict
    
image_names = ['building1.jpg', 'building2.jpg']
image_path = '../data/match'

# 1. 检测特征点
spd = SuperPointDetector()
sps = {}
for name in tqdm(sorted(image_names)):
    image_name = os.path.join(image_path, name)
    ret_dict = spd(cv2.imread(image_name))
    sps[name] = ret_dict

keypoints1 = sps[image_names[0]]['keypoints'] 
keypoints2 = sps[image_names[1]]['keypoints'] 
keypoints1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=20) for pt in keypoints1]
keypoints2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=20) for pt in keypoints2]
D1 = sps[image_names[0]]['descriptors'] * 1.0
D2 = sps[image_names[1]]['descriptors'] * 1.0

# 加载两张图像（都加载为彩色）
img1 = cv2.imread(image_path + '/' + image_names[0])  # 查询图像
img2 = cv2.imread(image_path + '/' + image_names[1])  # 目标图像（移除cv2.IMREAD_GRAYSCALE）

# 将BGR转换为RGB（OpenCV加载的是BGR，matplotlib需要RGB）
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 6. 绘制特征点匹配结果
image1_superpoint = cv2.drawKeypoints(img1_rgb,  # 使用RGB图像
                               keypoints1_cv, 
                               None, 
                               color=(0, 255, 0),  
                               flags=0)
image2_superpoint = cv2.drawKeypoints(img2_rgb,  # 使用RGB图像
                               keypoints2_cv, 
                               None, 
                               color=(0, 255, 0),  
                               flags=0)

# 7. 显示结果
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(image1_superpoint)  # 移除cmap='gray'
plt.axis("off")
plt.title('First Image with Keypoints')

plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(image2_superpoint)  # 移除cmap='gray'
plt.title('Second Image with Keypoints')

plt.show()