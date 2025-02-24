import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomCrop, Resize
from PIL import Image

'''
    读取一张高分辨率图像，并读取它对应的低分辨率图像
'''

class SetDataset(Dataset):
    def __init__(self, root_dir):
        super(SetDataset, self).__init__()
        self.root_dir = root_dir # C:/今今/dataset/set5/HR
        self.image_filenames = [] # 存放  (hr_filename,lr_filename)
        img_list = os.listdir(self.root_dir)
        for file in img_list:
            hr_image = os.path.join(self.root_dir,file) # C:/今今/dataset/set5/HR/baby.png
            lr_image = self.root_dir.replace('HR','LR_bicubic') # C:/今今/dataset/set5/LR_bicubic
            lr_image = os.path.join(lr_image,'X4') # C:/今今/dataset/set5/LR_bicubic/X4
            lr_image = os.path.join(lr_image,file) # C:/今今/dataset/set5/LR_bicubic/X4/baby.png
            base,extension = os.path.splitext(lr_image) # C:/今今/dataset/set5/LR_bicubic/X4/baby, .png
            base += 'X4' # C:/今今/dataset/set5/LR_bicubic/X4/babyX4
            lr_image = base+extension # C:/今今/dataset/set5/LR_bicubic/X4/baby.png
            self.image_filenames.append((hr_image,lr_image))

    def getData(self):
            return self.image_filenames



    def __getitem__(self, index):
        return self.image_filenames[index]
    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    # 使用示例
    dataset = SetDataset(root_dir='C:/今今/dataset/set5/HR')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # print(len(dataloader))
    # for (hr_images, lr_images) in dataloader:
    #     print(lr_images,hr_images)
    #     break
    imgs = dataset.getData()
    print(imgs)
    print(len(dataset))