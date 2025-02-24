import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 输入通道数为1，输出通道数为16，卷积核大小为3x3，步长为2，填充为1
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 输入通道数为16，输出通道数为32，卷积核大小为3x3，步长为2，填充为1
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # 输入通道数为32，输出通道数为64，卷积核大小为7x7
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # 输入通道数为64，输出通道数为32，卷积核大小为7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 输入通道数为32，输出通道数为16，卷积核大小为3x3，步长为2，填充为1，输出填充为1
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 输入通道数为16，输出通道数为1，卷积核大小为3x3，步长为2，填充为1，输出填充为1
            nn.Sigmoid()  # 输出层使用Sigmoid激活函数，将输出值限制在[0, 1]范围内
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
