import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomCrop, Resize
from PIL import Image

'''
    读取一张高分辨率，对他进行处理得到低分辨率
'''

class TrainDataset(Dataset):
    def __init__(self, root_dir):
        super(TrainDataset, self).__init__()
        self.root_dir = root_dir # C:/今今/dataset/DIV2K/train/DIV2K_train_HR
        self.image_filenames = [] # 存放  hr_filename
        self.transform_hr = RandomCrop(256) # 随机裁剪
        self.transform_lr = Resize(64)
        self.to_tensor = ToTensor()
        for _,_,files in os.walk(self.root_dir):
            for file in files:
                image_file = os.path.join(self.root_dir,file)
                self.image_filenames.append(image_file)



    def __getitem__(self, index):
        hr_filename = self.image_filenames[index]
        hr_image = Image.open(hr_filename)
        hr_image = self.to_tensor(hr_image)
        hr_image = self.transform_hr(hr_image) # 上述，进行降采样
        hr_image = hr_image.unsqueeze(0)
        lr_image = nn.functional.interpolate(hr_image, scale_factor=1 / 4, mode='bicubic') # 得到训练的lr图像
        hr_image, lr_image = hr_image.squeeze(0), lr_image.squeeze(0)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    # 使用示例
    dataset = TrainDataset(root_dir='C:/今今/dataset/DIV2K/train/DIV2K_train_HR')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataloader))
    for lr_images, hr_images in dataloader:
        print(hr_images)