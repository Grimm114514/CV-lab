"""
SuperPoint 特征点检测+特征匹配

迭代估计多个单应变换

完善函数：iterative_homography_estimation (注意参数：num_iterations, ransac_threshold)
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
        "keypoint_threshold": 0.004,
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

def iterative_homography_estimation(src_pts, dst_pts, good_matches, keypoints1_cv, keypoints2_cv, 
                                   img1_rgb, img2_rgb, num_iterations=3, ransac_threshold=3.0):
    """
    迭代估计多个单应变换
    
    参数:
        src_pts: 源图像中的匹配点
        dst_pts: 目标图像中的匹配点  
        good_matches: 好的匹配对
        keypoints1_cv, keypoints2_cv: 关键点
        img1_rgb, img2_rgb: 原始图像
        num_iterations: 迭代次数
        ransac_threshold: RANSAC重投影阈值
    """
    
    remaining_src_pts = src_pts.copy()
    remaining_dst_pts = dst_pts.copy()
    remaining_matches = good_matches.copy()
    
    homography_results = []
    
    for iteration in range(num_iterations):
        if len(remaining_src_pts) < 4:  # 至少需要4个点来计算单应矩阵
            break
            
        print(f"{iteration+1}: {len(remaining_src_pts)} feature matches")
        
        # 使用 RANSAC 计算单应性矩阵
        H, mask = cv2.findHomography(remaining_src_pts, remaining_dst_pts, 
                                   method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
        
        if H is None:
            break
            
        # 计算内点（满足当前单应变换的匹配点）
        inlier_mask = mask.ravel().astype(bool)
        inlier_count = np.sum(inlier_mask)
        
        print(f"{iteration+1}: find {inlier_count} inliers ({inlier_count/len(remaining_src_pts)*100:.2f}%)")
        
        # 提取当前迭代的内点
        current_inliers_src = remaining_src_pts[inlier_mask]
        current_inliers_dst = remaining_dst_pts[inlier_mask]
        current_inlier_matches = [remaining_matches[i] for i in range(len(remaining_matches)) if inlier_mask[i]]
        
        # 保存当前迭代的结果
        homography_results.append({
            'homography': H,
            'inlier_src_pts': current_inliers_src,
            'inlier_dst_pts': current_inliers_dst,
            'inlier_matches': current_inlier_matches,
            'inlier_mask': inlier_mask,
            'iteration': iteration + 1
        })
        
        # (修改) 从匹配点中移除当前内点，并绘制当前匹配结果。
    
    return homography_results

def plot_iteration_result(img1_rgb, img2_rgb, keypoints1_cv, keypoints2_cv, 
                         current_matches, result_dict, iteration):
    """绘制单次迭代的匹配结果"""
    
    # 为每次迭代使用不同的颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    match_color = colors[(iteration - 1) % len(colors)]
    
    # 创建匹配掩码（当前迭代的所有匹配都是内点）
    matches_mask = [1] * len(current_matches)
    
    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1_rgb, keypoints1_cv, img2_rgb, keypoints2_cv, 
                                 current_matches, None,
                                 matchColor=match_color,
                                 singlePointColor=match_color,
                                 matchesMask=matches_mask,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 添加文本信息
    cv2.putText(img_matches, f'Iteration {iteration}: {len(current_matches)} inliers', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, match_color, 2)
    
    # 使用 Matplotlib 显示结果
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Iteration {iteration} - Homography Estimation')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 主程序
image_names = ['homo1.jpg', 'homo2.jpg']
image_path = '../3-2/data/homo'

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

# 2. 使用 BFMatcher 进行特征匹配，寻找最近邻和次近邻
matches_nn = []
bf = cv2.BFMatcher(cv2.NORM_L2)
matches_nn = bf.match(D1, D2)
knn_matches = bf.knnMatch(D1, D2, k=2)

# 3. 应用比例测试 (Ratio Test)
good_matches = []
ratio_thresh = 0.7
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

print(f"# of matches: {len(good_matches)}")

# 3. 加载两张图像（都加载为彩色）
img1 = cv2.imread(image_path + '/' + image_names[0])
img2 = cv2.imread(image_path + '/' + image_names[1])

# 将BGR转换为RGB（OpenCV加载的是BGR，matplotlib需要RGB）
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 4. 提取匹配点
src_pts = np.float32([keypoints1_cv[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2_cv[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 5. 执行迭代单应变换估计
homography_results = iterative_homography_estimation(
    src_pts, dst_pts, good_matches, keypoints1_cv, keypoints2_cv, 
    img1_rgb, img2_rgb, num_iterations=4, ransac_threshold=2.0
)

# 6. 显示汇总结果
if homography_results:    
    for result in homography_results:
        print(f"{result['iteration']}: {len(result['inlier_matches'])} inliers")
else:
    print("No homographies ")

# 7. 显示原始特征点和匹配结果
plt.figure(figsize=(20, 10))

# 绘制特征点
image1_superpoint = cv2.drawKeypoints(img1_rgb, keypoints1_cv, None, color=(0, 255, 0), flags=0)
image2_superpoint = cv2.drawKeypoints(img2_rgb, keypoints2_cv, None, color=(0, 255, 0), flags=0)

plt.subplot(1, 2, 1)
plt.imshow(image1_superpoint)
plt.axis("off")
plt.title('First Image with Keypoints')

plt.subplot(1, 2, 2)
plt.imshow(image2_superpoint)
plt.axis("off")
plt.title('Second Image with Keypoints')

plt.tight_layout()
plt.show()

# 显示原始匹配结果
matching_result = cv2.drawMatches(img1_rgb, keypoints1_cv, img2_rgb, keypoints2_cv,
                                good_matches[:], None, 
                                matchColor=(0, 255, 255),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(20, 10))
plt.imshow(matching_result)
plt.axis("off")
plt.title('SuperPoint Matching with Ratio Test')
plt.tight_layout()
plt.show()