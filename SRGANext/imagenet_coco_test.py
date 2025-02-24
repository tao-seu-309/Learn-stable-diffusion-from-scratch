import torch
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from torchvision.transforms import ToTensor
import logging
logger = logging.getLogger(__name__)
import numpy as np
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
'''
  用于在ImageNet100，COCO2017上测试
  计算SSIM和PSNR
    加载模型
    先读入照片，然后转成Tensor格式，对数据处理，进行推理
    将输出转回numpy，乘上255后得到输出像素
    变换回[H,W,C]的维度
    和处理过的hr_img计算
'''


# 需要加载的模型和需要加载的数据集dataloader类型
from utils.ImageNet100_test_dataloader import ImageNetDataset
from utils.COCO_test_dataloader import COCODataset
from utils.origin_module import Generator
# from utils.SRGANext_module import Generator


# 测试集路径
root_dir = 'C:/今今/dataset/imagenet100'
# root_dir = 'C:/今今/dataset/COCO2017/test2017'


# 加载测试集,直接返回的就是[C,H,W]的Tensor
dataset = ImageNetDataset(root_dir)


# 加载模型
model = Generator()
model_dict = torch.load('modules/SRGAN_weight/generator_100.pth')
# model_dict = torch.load('modules/SRGANext_weight/generator_100.pth')
model.load_state_dict(model_dict)
model.eval()
model = model.to(device)
logging.info(f'共:{len(dataset)}个数据加载完毕，模型加载完毕')


sum_ssim,sum_psnr,i = 0,0,0
# 计算
for lr_img,hr_img in dataset:
    if lr_img.shape[0] != 3:
        continue
    if hr_img.shape[1]>700 or hr_img.shape[2]>700:
        continue
    if i>10000:
        break
    i += 1
    hr_img = hr_img.numpy()
    hr_img = hr_img.transpose(1,2,0)
    lr_img = lr_img.unsqueeze(0)
    lr_img = lr_img.to(device)
    with torch.no_grad():
        fake_img = model(lr_img)
    fake_img = fake_img.squeeze(0).cpu().detach().numpy()
    fake_img = fake_img.transpose(1,2,0)
    fake_img, hr_img = (fake_img * 255).round(), (hr_img * 255).round()
    psnr = PSNR(hr_img,fake_img,data_range=255)
    ssim = SSIM(hr_img,fake_img,channel_axis=2,data_range=255)
    sum_psnr += psnr
    sum_ssim += ssim
    logger.info("第{}个,PSNR: {:.2f}dB, SSIM: {:.4f}".format(i, psnr, ssim))
    # print("PSNR: {:.2f}dB, SSIM: {:.4f}".format(psnr, ssim))
logger.info('mean_psnr={:.4f}dB,mean_ssim={:.4f}'.format(sum_psnr / i, sum_ssim / i))

