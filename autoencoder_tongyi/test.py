import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Autoencoder
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据
])
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建模型实例并加载训练好的权重
model = Autoencoder()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# 创建保存结果的目录
if not os.path.exists('results'):
    os.makedirs('results')

# 测试模型
total_samples = len(test_loader.dataset)
sample_count = 0
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, desc="Testing", unit="sample")):
        if sample_count >= 20:
            break
        img, _ = data
        output = model(img)

        # 可视化原始图像和重建图像
        img = img.squeeze().numpy()
        output = output.squeeze().numpy()
        plt.figure(figsize=(4, 2))  # 调整图像大小
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image', fontsize=8)  # 调整字体大小
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(output, cmap='gray')
        plt.title('Reconstructed Image', fontsize=8)  # 调整字体大小
        plt.axis('off')

        # 保存图像
        plt.savefig(f'results/{i}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

        sample_count += 1

print('Testing finished and results saved.')

# plt.show()