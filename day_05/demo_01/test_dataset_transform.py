import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.ToTensor()
# dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) # 也可以

train_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform=dataset_transform, train=False, download=True)

# print(test_set[0])  # 图片数据类型，标签
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("dataset_log")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i);

writer.close()
