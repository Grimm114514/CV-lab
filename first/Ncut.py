"""
熟悉归一化图割 使用方法
"""

from skimage import data, segmentation, color
from skimage import graph
import cv2
import os

profile_dir = './dataset'
os.makedirs("./Ncutresults", exist_ok=True)

for file in os.listdir(profile_dir):

    # 读取图像
    image = cv2.imread(os.path.join(profile_dir, file))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 提取超像素
    labels1 = segmentation.slic(img, compactness=30, n_segments=200, start_label=1)

    # 利用平均颜色来输出超像素
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

    # 生成RAG图
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)

    # 利用平均颜色来输出图割结果
    out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    # 直接保存结果图像
    # 转换回BGR格式用于cv2保存
    out2_bgr = cv2.cvtColor((out2 * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join("./Ncutresults", f"ncut_{file}"), out2_bgr)