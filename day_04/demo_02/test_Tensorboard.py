from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

# image_path = "dataset/train/ants_image/5650366_e22b7e1065.jpg"
image_path = "dataset/train/bees/21399619_3e61e5bb6f.jpg"

img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)  # 转换为ndarray

print(type(img_array))
print(img_array.shape)

# writer.add_image("ants", img_array, 2,dataformats="HWC") # 转换通道为HWC模式
writer.add_image("bees", img_array, 2, dataformats="HWC")  # 转换通道为HWC模式

for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

    #  tensorboard --logdir=logs --port=自定义

writer.close()
