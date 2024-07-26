import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)

dataloader_data = DataLoader(test_data, batch_size=64, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = Linear(196608, 10)  # 输入大小，输出大小

    def forward(self, input):
        output = self.linear1(input);
        return output


model = Model()

writer = SummaryWriter("linear_log")

step = 0;
for data in dataloader_data:
    imgs, targets = data
    # print(imgs.shape)
    writer.add_images("before_linear_img", imgs, step)
    imgs_reshape = torch.reshape(imgs, (1, 1, 1, -1))
    # imgs_reshape = torch.flatten(imgs) # 用这个得变通道
    # print(output.shape)
    output = model(imgs_reshape)
    writer.add_images("after_linear_img", output, step)
    step = step + 1

writer.close()
