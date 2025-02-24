import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Autoencoder

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据
])
train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 创建模型实例
model = Autoencoder()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        optimizer.zero_grad()  # 梯度清零
        output = model(img)  # 前向传播
        loss = criterion(output, img)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'autoencoder.pth')
print('Training finished and model saved.')
