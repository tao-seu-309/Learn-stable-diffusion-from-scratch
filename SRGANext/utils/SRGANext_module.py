import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    '''
        [b,c,h,w] --> permute [b,h,w,c] --> LN --> permute [b,c,h,w]
    '''
    def __init__(self,dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        return x





class SRGANextBlock(nn.Module):
    def __init__(self,dim, layer_scale_init_value=1e-6):
        super(SRGANextBlock, self).__init__()
        self.dconv1 = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=7,stride=1,padding=3,groups=dim)
        self.norm1 = LayerNorm(dim)
        self.pconv1 = nn.Conv2d(dim,4*dim,kernel_size=1,stride=1)
        self.act = nn.GELU()
        self.pconv2 = nn.Conv2d(4*dim,dim,kernel_size=1,stride=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self,x):
        shortcut = x
        x = self.dconv1(x)
        x = self.norm1(x)
        x = self.pconv1(x)
        x = self.act(x)
        x = self.pconv2(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + x
        return x




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.stem = nn.Conv2d(3,64,kernel_size=7,stride=1,padding=3)
        self.norm1 = LayerNorm(64)
        self.block1 = nn.Sequential(*[SRGANextBlock(64) for _ in range(3)])
        self.norm2 = LayerNorm(64)
        self.pconv1 = nn.Conv2d(64,128,kernel_size=1,stride=1)
        self.block2 = nn.Sequential(*[SRGANextBlock(128) for _ in range(3)])
        self.norm3 = LayerNorm(128)
        self.pconv2 = nn.Conv2d(128,256,kernel_size=1,stride=1)
        self.block3 = nn.Sequential(*[SRGANextBlock(256) for _ in range(9)])
        self.norm4 = LayerNorm(256)
        self.pconv3 = nn.Conv2d(256,512,kernel_size=1,stride=1)
        self.block4 = nn.Sequential(*[SRGANextBlock(512) for _ in range(3)])
        self.upsample = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv1 = nn.Conv2d(128,3,kernel_size=7,stride=1,padding=3)
    def forward(self,x):
        x = self.stem(x)
        x = self.norm1(x)
        x = self.block1(x)
        x = self.norm2(x)
        x = self.pconv1(x)
        x = self.block2(x)
        x = self.norm3(x)
        x = self.pconv2(x)
        x = self.block3(x)
        x = self.norm4(x)
        x = self.pconv3(x)
        x = self.block4(x)
        x = self.upsample(x)
        x = self.conv1(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,96,kernel_size=7,stride=2,padding=3),
            LayerNorm(96)
        )
        self.block1 = nn.Sequential(*[SRGANextBlock(96) for _ in range(3)])
        self.downsample1 = nn.Sequential(
            LayerNorm(96),
            nn.Conv2d(96,192,kernel_size=2,stride=2)
        )
        self.block2 = nn.Sequential(*[SRGANextBlock(192) for _ in range(3)])
        self.downsample2 = nn.Sequential(
            LayerNorm(192),
            nn.Conv2d(192, 384, kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(*[SRGANextBlock(384) for _ in range(9)])
        self.downsample3 = nn.Sequential(
            LayerNorm(384),
            nn.Conv2d(384, 768, kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(*[SRGANextBlock(768) for _ in range(3)])
        self.endLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.stem(x)
        shortcut = x
        x = shortcut+self.block1(x)
        x = self.downsample1(x)
        shortcut = x
        x = shortcut+self.block2(x)
        x = self.downsample2(x)
        shortcut = x
        x = shortcut+self.block3(x)
        x = self.downsample3(x)
        shortcut = x
        x = shortcut+self.block4(x)
        x = self.endLayer(x)
        return x

