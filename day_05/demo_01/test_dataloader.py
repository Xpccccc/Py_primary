import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# 每个test_dataloader数据 包含 batch_size 个test_data的数据

# 测试数据集中的第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader_log")

for epoch in range(2):
    step = 0;
    for data in test_dataloader:
        imgs, targets = data
        # print(type(imgs))
        # print(imgs.shape)
        # print(targets)
        writer.add_images("epoch : {}".format(epoch), imgs, step)  # 注意，这里是images，因为是四维的tensor，也就是多张图片
        step = step + 1

writer.close()
