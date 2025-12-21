"""
NetVLAD 

采用VGG16 pretrained_model

如果利用其他预训练模型，请参考hloc中，pipeline_netvlad.ipynb
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 1. 添加 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义NetVLAD层
class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=512, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 100.0  # 初始化聚类中心的系数
        self.normalize_input = normalize_input
        
        # 初始化聚类中心
        self.clsts = nn.Parameter(torch.rand(num_clusters, dim))
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        
    def forward(self, x):
        N, C, H, W = x.shape  # N=batch_size, C=channels, H=height, W=width
        
        if self.normalize_input:
            x = x / torch.norm(x, p=2, dim=1, keepdim=True)  # L2归一化
        
        soft_assign = self.conv(x)  # [N, num_clusters, H, W]
        soft_assign = soft_assign.view(N, self.num_clusters, -1)
        soft_assign = nn.functional.softmax(soft_assign, dim=1)  # [N, num_clusters, H*W]
        
        x_flatten = x.view(N, C, -1)  # [N, C, H*W]
        residuals = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                    self.clsts.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0)  # [N, num_clusters, C, H*W]
        residuals = residuals * soft_assign.unsqueeze(2)  # [N, num_clusters, C, H*W]
        
        vlad = residuals.sum(dim=-1)  # [N, num_clusters, C]
        vlad = vlad / torch.norm(vlad, p=2, dim=2, keepdim=True)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # [N, num_clusters * C]
        vlad = vlad / torch.norm(vlad, p=2, dim=1, keepdim=True)  # L2归一化
        
        return vlad

# 加载预训练的VGG16模型用于特征提取
def extract_vgg16_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0).to(device)  # 移动到 device
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:-2]
    vgg16 = vgg16.to(device)
    vgg16.eval()

    with torch.no_grad():
        features = vgg16(image)

    return features

# 加载NetVLAD预训练权重
def load_pretrained_netvlad(netvlad, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netvlad.load_state_dict(checkpoint['state_dict'], strict=False)
    netvlad.to(device)

# 提取NetVLAD特征描述符
def extract_netvlad_descriptor(image_path, netvlad):
    features = extract_vgg16_features(image_path)  # 已在 device 上
    netvlad.eval()
    with torch.no_grad():
        vlad_descriptor = netvlad(features)
    return vlad_descriptor.detach().cpu().numpy()  # 移回 CPU 供 sklearn 使用

# 最近邻图像检索
def image_retrieval(query_image_path, dataset_folder, netvlad, top_n=5):
    query_descriptor = extract_netvlad_descriptor(query_image_path, netvlad)
    similarities = []
    
    for filename in os.listdir(dataset_folder):
        # 过滤非图像文件
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        image_path = os.path.join(dataset_folder, filename)
        try:
            vlad_descriptor = extract_netvlad_descriptor(image_path, netvlad)
            similarity = cosine_similarity(query_descriptor, vlad_descriptor)[0][0]
            similarities.append((filename, similarity))
        except (ValueError, Exception) as e:
            print(f"跳过图像 {filename}: {e}")
            continue
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# 显示检索结果
def show_results(query_image_path, similar_images, dataset_folder):
    plt.figure(figsize=(15, 5))  # 调整大小以适应更多图像
    query_image = cv2.imread(query_image_path)
    plt.subplot(1, len(similar_images) + 1, 1)
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.title('Query Image')
    plt.axis('off')

    for i, (filename, similarity) in enumerate(similar_images):
        image_path = os.path.join(dataset_folder, filename)
        similar_image = cv2.imread(image_path)
        plt.subplot(1, len(similar_images) + 1, i + 2)
        plt.imshow(cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB))
        plt.title(f'{filename}')
        plt.axis('off')

    plt.show()

# 主函数
def main():
    dataset_folder = './data/vlad/vlad1/mapping'
    query_image_path = './data/vlad/vlad1/query/query1.jpg'  # 替换为你的查询图像路径
    
    num_clusters = 128
    netvlad = NetVLAD(num_clusters=num_clusters, dim=512)
    
    # 加载预训练NetVLAD权重
    checkpoint_path = 'netvlad_pretrained.pth.tar'  # 替换为实际的路径
    load_pretrained_netvlad(netvlad, checkpoint_path)  # 内部会把 netvlad.to(device)
    
    netvlad.eval()  # 设置为评估模式
    
    # 检索并显示结果
    similar_images = image_retrieval(query_image_path, dataset_folder, netvlad, top_n=5)
    show_results(query_image_path, similar_images, dataset_folder)

if __name__ == "__main__":
    main()
