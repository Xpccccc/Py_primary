import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
dataloader_data = DataLoader(test_data, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)  # 向上取整

    def forward(self, x):
        x = self.maxpool1(x)
        return x


model = Model()

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]]);
# input = torch.reshape(input, (-1, 1, 5, 5))
#
# output = model(input)
# print(output)

writer = SummaryWriter("maxpool_logs")
step = 0;
for data in dataloader_data:
    imgs, targets = data
    writer.add_images("before_maxpool", imgs, step)
    output = model(imgs)  # 池化不改变维度
    writer.add_images("after_maxpool", output, step)
    step = step + 1

writer.close()
