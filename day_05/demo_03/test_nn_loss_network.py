import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)

dataloader_data = DataLoader(test_data, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = Sequential(  # 顺序执行下面模型
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = Model()

sft = nn.Softmax()
loss = nn.CrossEntropyLoss()
step = 0;
for data in dataloader_data:
    imgs, targets = data
    outputs = model(imgs)
    outputs = sft(outputs)
    result_loss = loss(outputs, targets)
    # result_loss.backward()
    print(result_loss)
