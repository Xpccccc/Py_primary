import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np

image_path = "dataset/train/ants_image/6240338_93729615ec.jpg"
img_PIL = Image.open(image_path)

img_array = np.array(img_PIL)

print(img_array.shape)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_PIL)

# print(tensor_img)
print(tensor_img.shape)

writer.add_image("Tensor_img", tensor_img)  # 通道数纬度匹配

writer.close()