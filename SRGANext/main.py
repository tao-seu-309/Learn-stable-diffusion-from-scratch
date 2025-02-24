import torch
import torchvision.transforms as transforms
from PIL import Image
# from utils.SRGANext_module import Generator
from utils.origin_module import Generator
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
'''
    将一张图片加载进来，并得到超分辨率后的图像
'''
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


# 加载预训练的SRGAN模型
model = Generator()
# model_dict = torch.load('modules/SRGANext_weight/generator_100.pth')
model_dict = torch.load('modules/SRGAN_weight/generator_100.pth')
model.load_state_dict(model_dict)
model.eval()
model = model.to(device)
logger.info(f"生成器模型加载完毕")

# 定义图像变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载测试图像
img = Image.open('test_image/0001x4.png')

# 对测试图像进行预处理
img = transform(img)
img = img.unsqueeze(0)
img = img.to(device)

# 将测试图像输入SRGAN模型进行超分辨率重建
with torch.no_grad():
    output = model(img)

# 将输出图像转换为PIL.Image对象并保存
output = output.squeeze(0).cpu().detach().numpy()
output = (output + 1) / 2.0 * 255.0
output = output.clip(0, 255).astype('uint8')
output = Image.fromarray(output.transpose(1, 2, 0))
output.save('output/main/main_25220.png')

