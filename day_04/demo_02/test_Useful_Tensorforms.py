import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

image_path = "dataset/train/ants_image/6240338_93729615ec.jpg"
img_PIL = Image.open(image_path)

writer = SummaryWriter("logs")

# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_PIL)

writer.add_image("Tensor_img", tensor_img)  # 通道数纬度匹配

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 1], [0.5, 0.5, 4]);
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("Normalize_img", img_norm, 5)  # 通道数纬度匹配

# Resize
print(type(img_PIL))
print(img_PIL.size)  # 这只会打出高和宽，不会打出通道数
trans_resize = transforms.Resize((512, 512))
# trans_resize = transforms.Resize(512) # (size * height / width, size).,相当于以以最小边等比例放大
img_resize = trans_resize(img_PIL)
print(img_resize.size)

img_resize = tensor_trans(img_resize)

writer.add_image("Resize_img", img_resize, 0)  # 通道数纬度匹配

print(type(img_resize))

# Compose -- Resize , ToTensor ,结合transforms里面的多个函数
trans_resize2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize2, tensor_trans])  # 将更改好的图片转换为tensor
img_resize2 = trans_compose(img_PIL)

writer.add_image("Resize_img", img_resize2, 1)  # 通道数纬度匹配
print(type(img_resize2))

# RandomCrop
# trans_crop = transforms.RandomCrop((100,200))
trans_crop = transforms.RandomCrop(100)  # 及正方形(100*100)
trans_compose2 = transforms.Compose([trans_crop, tensor_trans])  # 将裁减好的图片转换为tensor
for i in range(10):
    img_crop = trans_compose2(img_PIL)
    writer.add_image("Crop_img", img_crop, i)  # 通道数纬度匹配

writer.close()
