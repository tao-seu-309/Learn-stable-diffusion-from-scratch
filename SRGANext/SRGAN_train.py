import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.utils import save_image
from utils.origin_module import Generator, Discriminator
from utils.train_dataloader import TrainDataset
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device:{device}")

# 定义超参数
batch_size = 2
lr = 1e-4
num_epochs = 50
scale_factor = 2
# root_dir = 'C:/今今/dataset/DIV2K/train/DIV2K_train_HR'
root_dir = 'data/DIV2K/train/DIV2K_train_HR'



# 加载数据集
train_dataset = TrainDataset(root_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
logger.info(f"数据集加载完毕，共{len(train_loader)}组")

# 定义生成器和判别器模型

generator = Generator(scale_factor=scale_factor).to(device)
discriminator = Discriminator().to(device)
generator_dict = torch.load('modules/SRGAN_weight/generator_100.pth')
# discriminator_dict = torch.load('modules/discriminator_100.pth')
generator.load_state_dict(generator_dict)
# discriminator.load_state_dict(discriminator_dict)
logger.info(f"生成器，辨别器权重加载完毕")

# 定义损失函数和优化器
content_loss = nn.MSELoss().to(device) # 均方误差
adversarial_loss = nn.BCELoss().to(device) # 二分类交叉熵
# generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
# discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)


# 加载VGG19模型
vgg19 = models.vgg19(pretrained=True).features
# 冻结VGG19模型的权重
for param in vgg19.parameters():
    param.requires_grad = False
# 定义VGG19模型的损失函数
content_loss_vgg = nn.MSELoss()
if torch.cuda.is_available():
    vgg19.to(device)
    content_loss_vgg.to(device)
logger.info(f"VGG权重加载完毕")

# 训练模型
for epoch in range(num_epochs):
    for i, (lr_images, hr_images) in enumerate(train_loader):
        # 将数据移动到GPU上
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # 计算生成器损失
        generated_high_res = generator(lr_images) # 生成器生成
        content_loss_value = content_loss(hr_images, generated_high_res) # 均方误差 高分辨率 vs 生成高分辨率
        adversarial_loss_value = adversarial_loss(discriminator(generated_high_res), torch.ones(batch_size, 1).to(device)) # 二分类交叉熵
        generator_loss = content_loss_value + 1e-3 * adversarial_loss_value

        # 计算VGG19损失
        high_res_vgg = vgg19(hr_images)
        generated_high_res_vgg = vgg19(generated_high_res)
        content_loss_vgg_value = content_loss_vgg(high_res_vgg, generated_high_res_vgg)
        generator_loss += content_loss_vgg_value
        generator_loss /= 2

        # 更新生成器权重
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # 计算判别器损失
        dis_hr = discriminator(hr_images)
        dis_generated = discriminator(generated_high_res.detach()).detach()
        # # 软标签版
        # real_loss = adversarial_loss(dis_hr,
        #                              (torch.ones(batch_size, 1) - 1e-3 * torch.rand(batch_size, 1)).to(device))
        # fake_loss = adversarial_loss(dis_generated,
        #                              (torch.zeros(batch_size, 1) + 1e-3 * torch.rand(batch_size, 1)).to(device))


        # 非软标签版
        real_loss = adversarial_loss(discriminator(hr_images),
                                     torch.ones(batch_size, 1).to(device))
        fake_loss = adversarial_loss(discriminator(generated_high_res.detach()),
                                     torch.zeros(batch_size, 1).to(device))

        discriminator_loss = (real_loss + fake_loss) / 2
        # 更新判别器权重
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # 输出训练信息
        if (i + 1) % 50 == 0:
            logger.info(f"Training epoch {epoch+1}/{num_epochs}, num_steps {i+1}/{len(train_loader)},"
                        f" Generator Loss: {generator_loss.item():.4f}, Discriminator Loss:{discriminator_loss.item():.4f}，"
                        f" Real Loss:{real_loss.item():.4f}, Fake Loss:{fake_loss.item():.4f}")
            logger.info(f"{dis_hr},{dis_generated}")

            # 保存模型和生成的图像
    with torch.no_grad():
        generator.eval()
        hr_images = next(iter(train_loader))[1][:8].to(device)
        lr_images = nn.functional.interpolate(hr_images, scale_factor=1/scale_factor, mode='bicubic')
        fake_hr_images = generator(lr_images)
        save_image(hr_images, 'output/SRGAN/hr_images_{}.png'.format(epoch + 1))
        save_image(lr_images, 'output/SRGAN/lr_images_{}.png'.format(epoch + 1))
        save_image(fake_hr_images, 'output/SRGAN/fake_hr_images_{}.png'.format(epoch + 1))
        if (epoch+1) % 10 == 0:
            torch.save(generator.state_dict(), 'modules/SRGAN_weight/generator_{}.pth'.format(epoch + 1))
            torch.save(discriminator.state_dict(), 'modules/SRGAN_weight/discriminator_{}.pth'.format(epoch + 1))