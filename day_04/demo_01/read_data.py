import os.path

from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):  # 初始化路径
        self.root_dir = root_dir
        self.lable_dir = label_dir
        self.path = os.path.join(self.root_dir, self.lable_dir)
        self.img_path = os.listdir(self.path)  # 图片列表

    def __getitem__(self, idx):  # []可以访问
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.lable_dir, img_name)
        img = Image.open(img_item_path)
        lable = self.lable_dir
        return img, lable

    def __len__(self):  # 数据长度
        return len(self.img_path);


root_dir = "dataset/train"
ants_lable_dir = "ants_image"
bees_lable_dir = "bees"

ants_dataset = MyData(root_dir, ants_lable_dir)
bees_dataset = MyData(root_dir, bees_lable_dir)

train_dataset = ants_dataset + bees_dataset
