import sys
import os
sys.path.append('./SuperPoint')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 从match.py导入NetVLAD相关函数和类
from match import (
    NetVLAD,
    extract_netvlad_descriptor,
    load_pretrained_netvlad,
    image_retrieval,
    device
)

# 从SuperPoint模块导入
from SuperPoint.superpoint import SuperPoint
from SuperPoint.utils import image2tensor
import torch


# ==================== SuperPoint 特征匹配 ====================
class SuperPointDetector(object):
    """SuperPoint特征检测器"""
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
        self.config = {**self.default_config, **config}
        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": [image.shape[0], image.shape[1]],
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }

        return ret_dict


def match_superpoint(query_image_path, retrieved_image_path, ratio_thresh=0.75):
    """使用SuperPoint进行特征匹配（Glue类方法 - Ratio Test）"""
    spd = SuperPointDetector()

    # 检查查询图像路径
    if not os.path.exists(query_image_path):
        raise FileNotFoundError(f"查询图像路径不存在: {query_image_path}")
    if not os.path.exists(retrieved_image_path):
        raise FileNotFoundError(f"检索图像路径不存在: {retrieved_image_path}")

    img1 = cv2.imread(query_image_path)
    img2 = cv2.imread(retrieved_image_path)

    # 检查图像是否成功加载
    if img1 is None:
        raise ValueError(f"无法加载查询图像: {query_image_path}")
    if img2 is None:
        raise ValueError(f"无法加载检索图像: {retrieved_image_path}")

    ret1 = spd(img1)
    ret2 = spd(img2)

    keypoints1 = ret1['keypoints']
    keypoints2 = ret2['keypoints']
    D1 = ret1['descriptors'] * 1.0
    D2 = ret2['descriptors'] * 1.0

    keypoints1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=20) for pt in keypoints1]
    keypoints2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=20) for pt in keypoints2]

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(D1, D2, k=2)

    # Glue类方法：比例测试 (Ratio Test)
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return img1, img2, keypoints1_cv, keypoints2_cv, good_matches


def match_sift(query_image_path, retrieved_image_path, ratio_thresh=0.75):
    """使用SIFT进行特征匹配（Glue类方法 - Ratio Test）"""
    # 检查查询图像路径
    if not os.path.exists(query_image_path):
        raise FileNotFoundError(f"查询图像路径不存在: {query_image_path}")
    if not os.path.exists(retrieved_image_path):
        raise FileNotFoundError(f"检索图像路径不存在: {retrieved_image_path}")

    img1 = cv2.imread(query_image_path)
    img2 = cv2.imread(retrieved_image_path)

    # 检查图像是否成功加载
    if img1 is None:
        raise ValueError(f"无法加载查询图像: {query_image_path}")
    if img2 is None:
        raise ValueError(f"无法加载检索图像: {retrieved_image_path}")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Glue类方法：比例测试 (Ratio Test)
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return img1, img2, keypoints1, keypoints2, good_matches


def visualize_matches(query_image_path, retrieved_images, dataset_folder, use_superpoint=True):
    """可视化匹配结果"""
    n_images = len(retrieved_images)
    fig = plt.figure(figsize=(20, 5 * n_images))
    
    for idx, (filename, similarity) in enumerate(retrieved_images):
        image_path = os.path.join(dataset_folder, filename)
        print(f"\n匹配 Top-{idx+1}: {filename} (相似度: {similarity:.4f})")
        
        if use_superpoint:
            img1, img2, kp1, kp2, matches = match_superpoint(query_image_path, image_path)
            match_type = "SuperPoint"
        else:
            img1, img2, kp1, kp2, matches = match_sift(query_image_path, image_path)
            match_type = "SIFT"
        
        print(f"  匹配点数量: {len(matches)}")
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        matching_result = cv2.drawMatches(
            img1_rgb, kp1, 
            img2_rgb, kp2,
            matches[:100],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.subplot(n_images, 1, idx + 1)
        plt.imshow(matching_result)
        plt.title(f'Top-{idx+1}: {filename} | 相似度: {similarity:.4f} | {match_type}匹配点: {len(matches)}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 配置路径
    dataset_folder = './data/retri2/sources'
    query_folder = './data/retri2/mapping'
    checkpoint_path = 'netvlad_pretrained.pth.tar'
    
    if not os.path.exists(dataset_folder):
        print(f"错误: 数据集文件夹 '{dataset_folder}' 不存在！")
        print("请创建数据文件夹并放入图像，或修改 dataset_folder 路径")
        return
    
    query_images = [f for f in os.listdir(query_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not query_images:
        print(f"错误: 在 '{query_folder}' 中未找到图像文件！")
        return
    
    print(f"找到 {len(query_images)} 张查询图像")
    
    # 初始化NetVLAD（从match.py导入）
    print("\n初始化NetVLAD模型...")
    num_clusters = 128
    netvlad = NetVLAD(num_clusters=num_clusters, dim=512)
    
    if os.path.exists(checkpoint_path):
        print(f"加载预训练权重: {checkpoint_path}")
        load_pretrained_netvlad(netvlad, checkpoint_path)
    else:
        print(f"警告: 未找到预训练权重文件 '{checkpoint_path}'，使用随机初始化")
    
    netvlad.eval()
    
    # 选择特征匹配方法
    use_superpoint = True  # True: 使用SuperPoint, False: 使用SIFT
    match_method = "SuperPoint" if use_superpoint else "SIFT"
    print(f"\n使用特征匹配方法: {match_method}")
    
    # 对每张查询图像进行检索和匹配
    for query_image_name in query_images[:3]:
        query_image_path = os.path.join(query_folder, query_image_name)
        
        print(f"\n{'='*60}")
        print(f"处理查询图像: {query_image_name}")
        print(f"{'='*60}")
        
        # NetVLAD检索Top-3（使用从match.py导入的函数）
        print("\n执行NetVLAD图像检索...")
        retrieved_images = image_retrieval(query_image_path, dataset_folder, netvlad, top_n=3)
        
        print("\nTop-3检索结果:")
        for idx, (filename, similarity) in enumerate(retrieved_images):
            print(f"  Top-{idx+1}: {filename} (相似度: {similarity:.4f})")
        
        # 特征匹配和可视化
        print(f"\n开始{match_method}特征匹配...")
        visualize_matches(query_image_path, retrieved_images, dataset_folder, use_superpoint=use_superpoint)
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()