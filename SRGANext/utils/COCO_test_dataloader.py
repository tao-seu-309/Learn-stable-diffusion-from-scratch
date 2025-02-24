import os
import torch
import torch.nn as nn
import torchvision.utils
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomCrop, Resize
from PIL import Image

'''
    低分辨率图像需要由高分辨率图像处理得到的测试集的dataloader
    通过nn.functional.interpolate(hr_image, scale_factor=1 / 4, mode='bicubic')得到
    返回两个Tensor类型
'''



class COCODataset(Dataset):
    def __init__(self, root_dir):
        super(COCODataset, self).__init__()
        self.root_dir = root_dir # C:/今今/dataset/COCO2017/test2017
        self.image_filenames = [] # 存放  hr_filename
        self.to_tensor = ToTensor()
        filelist = os.listdir(self.root_dir)
        for file in filelist:
            img_path = self.root_dir+'/'+file # C:/今今/dataset/COCO2017/test2017/00000000001.jpg
            self.image_filenames.append(img_path)


    def __getitem__(self, index):
        hr_filename = self.image_filenames[index]
        hr_image = Image.open(hr_filename)
        hr_image = self.to_tensor(hr_image)
        resize = Resize((hr_image.shape[1]-hr_image.shape[1]%4,hr_image.shape[2]-hr_image.shape[2]%4))
        hr_image = resize(hr_image)
        hr_image = hr_image.unsqueeze(0)
        lr_image = nn.functional.interpolate(hr_image, scale_factor=1 / 4, mode='bicubic')
        hr_image,lr_image = hr_image.squeeze(0),lr_image.squeeze(0)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    # 使用示例
    dataset = TestDataset(root_dir='C:/今今/dataset/COCO2017/test2017')
    for i in range(len(dataset)):
        lr_image,hr_image = dataset[i]
        print(lr_image.shape)
        print(hr_image.shape)
        print(type(lr_image))
        print('---------------------------------')