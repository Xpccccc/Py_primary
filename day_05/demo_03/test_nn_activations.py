import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)

dataloader_data = DataLoader(test_data, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input);
        output = self.sigmoid1(input);
        return output


model = Model()

# input = torch.tensor([[1, -1.1],
#                       [2, -0.1]])
# output = model(input)
# print(output)

# writer = SummaryWriter("relu_log")
writer = SummaryWriter("sigmoid_log")

step = 0;
for data in dataloader_data:
    imgs, targets = data
    writer.add_images("before_relu_img", imgs, step)
    output = model(imgs)
    writer.add_images("after_relu_img", output, step)
    step = step + 1

writer.close()
