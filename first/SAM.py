import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import numpy as np

# 添加 SAM 路径
sys.path.append('./RunSAM/segment-anything-main')

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# 创建正确的输出文件夹
os.makedirs("./SAMresults-vit_h", exist_ok=True)

# 检查 GPU 状态
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载SAM模型
sam_checkpoint = "./RunSAM/segment-anything-main/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# 加载SAM模型
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 创建自动掩码生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 处理 ./dataset 目录下的所有图像
image_dir = "./dataset"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for image_file in image_files:
    print(f"Processing {image_file}...")
    
    # 加载图像
    img_path = os.path.join(image_dir, image_file)
    image = Image.open(img_path).convert("RGB")
    image_array = np.array(image)
    
    # 使用SAM生成掩码
    masks = mask_generator.generate(image_array)
    
    # 创建分割可视化
    plt.figure(figsize=(15, 5))
    
    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.axis('off')
    
    # 所有掩码
    plt.subplot(1, 3, 2)
    plt.imshow(image_array)
    if len(masks) > 0:
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        for mask in sorted_masks:
            m = mask['segmentation']
            color = np.random.random(3)
            plt.imshow(np.dstack([m, m, m]) * color.reshape(1, 1, 3), alpha=0.5)
    plt.title(f"All Masks ({len(masks)} segments)")
    plt.axis('off')
    
    # 最大的几个掩码
    plt.subplot(1, 3, 3)
    plt.imshow(image_array)
    if len(masks) > 0:
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        for i, mask in enumerate(sorted_masks[:5]):  # 显示最大的5个
            m = mask['segmentation']
            color = plt.cm.tab10(i)[:3]
            plt.imshow(np.dstack([m, m, m]) * np.array(color).reshape(1, 1, 3), alpha=0.6)
    plt.title("Top 5 Largest Segments")
    plt.axis('off')
    
    # 保存结果 - 使用 PNG 格式避免 JPEG 的问题
    result_path = os.path.join("./SAMresults-vit_h", f"sam_{os.path.splitext(image_file)[0]}.png")
    plt.tight_layout()
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved result to {result_path}")

print("Processing completed!")