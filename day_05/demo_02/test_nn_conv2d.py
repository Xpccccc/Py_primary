import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载数据 测试数据比较少，所以下载测试数据
test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 处理数据
dataloader_data = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1,
                            padding=1)  # 输入3个通道，输出6个通道，说明中间有2个卷积核

    def forward(self, x):
        x = self.conv1(x)
        return x;


model = Model()

writer = SummaryWriter("logs");

step = 0
for data in dataloader_data:
    imgs, targets = data
    writer.add_images("before_conv2d_image", imgs, step)
    output = model(imgs)
    output = torch.reshape(output, (-1, 3, 32, 32))
    writer.add_images("after_conv2d_image", output, step) # 彩色图片rgb三通道
    step = step + 1;

writer.close()
