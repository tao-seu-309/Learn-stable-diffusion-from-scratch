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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
'''
  用于在DIV2K，Flickr，set5，set14上测试
  计算SSIM和PSNR
    加载模型
    先读入照片，然后转成Tensor格式，对数据处理，进行推理
    将输出转回numpy，乘上255后得到输出像素
    变换回[H,W,C]的维度
    和处理过的hr_img计算
'''


# 需要加载的模型和需要加载的数据集dataloader类型
from utils.Flickr_test_dataloader import FlickrDataset
from utils.DIV2K_test_dataloader import DIV2KDataset
from utils.set_test_dataloader import SetDataset
# from utils.origin_module import Generator
from utils.SRGANext_module import Generator


# 测试集路径
# root_dir = 'C:/今今/dataset/Flickr2K/Flickr2K_HR'
root_dir = 'C:/今今/dataset/DIV2K/valid/DIV2K_valid_HR'
# root_dir='C:/今今/dataset/set5/HR'
# root_dir='C:/今今/dataset/set14/HR'

# 加载测试集
# dataset = FlickrDataset(root_dir)
dataset = DIV2KDataset(root_dir)
# dataset = SetDataset(root_dir)
transform = ToTensor()

# 加载模型，加载数据
imgs_file = dataset.getData() # [[hr1,lr1],[hr2,lr2],...,[hrn,lrn]]
model = Generator()
# model_dict = torch.load('modules/SRGAN_weight/generator_100.pth')
model_dict = torch.load('modules/SRGANext_weight/generator_100.pth')
model.load_state_dict(model_dict)
model.eval()
model = model.to(device)
logging.info(f'共:{len(dataset)}个数据加载完毕，模型加载完毕')


sum_ssim,sum_psnr,i = 0,0,0
# 计算
for (hr_path,lr_path) in imgs_file:
    # if i==500:
    #     break
    i += 1
    hr_img,lr_img = Image.open(hr_path).convert('RGB'),Image.open(lr_path).convert('RGB')
    hr_img = transform(hr_img).numpy()
    if hr_img.shape[0] != 3:
        i -= 1
        continue
    hr_img = hr_img.transpose(1,2,0) # [C,H,W]--->[H,W,C]
    lr_img = transform(lr_img)
    lr_img = lr_img.unsqueeze(0)
    lr_img = lr_img.to(device)
    with torch.no_grad():
        fake_img = model(lr_img)
    fake_img = fake_img.squeeze(0).cpu().detach().numpy()
    fake_img = fake_img.transpose(1,2,0) # [C,H,W]--->[H,W,C]
    fake_img, hr_img = (fake_img * 255).round(), (hr_img * 255).round()
    if fake_img.shape != hr_img.shape:
        i -= 1
        continue
    psnr = PSNR(hr_img,fake_img,data_range=255)
    ssim = SSIM(hr_img,fake_img,channel_axis=2,data_range=255)
    sum_psnr += psnr
    sum_ssim += ssim
    logger.info("第{}个,PSNR: {:.2f}dB, SSIM: {:.4f}".format(i,psnr, ssim))
    # print("PSNR: {:.2f}dB, SSIM: {:.4f}".format(psnr, ssim))
logger.info('mean_psnr={:.4f}dB,mean_ssim={:.4f}'.format(sum_psnr/i,sum_ssim/i))

