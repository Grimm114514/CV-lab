import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

os.makedirs("./SAMresults", exist_ok=True)
# 选择预训练的DeepLabV3模型
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# 图像预处理函数
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 用ImageNet的均值和标准差进行归一化
])

# 使用GPU加速（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 处理 ./dataset 目录下的所有图像
image_dir = "./dataset"
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

for image_file in image_files:
    # 加载图像
    img_path = os.path.join(image_dir, image_file)
    img = Image.open(img_path)

    # 进行图像预处理
    input_tensor = transform(img).unsqueeze(0).to(device)  # 增加一个batch维度并移动到设备

    # 进行预测
    with torch.no_grad():
        output = model(input_tensor)["out"][0]  # 获取预测结果
        output_predictions = output.argmax(0)  # 获取每个像素的类别预测

    # 保存分割结果
    result_path = os.path.join("./SAMresults", f"segmented_{image_file}")
    plt.imsave(result_path, output_predictions.cpu().numpy())