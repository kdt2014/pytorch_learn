import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

# 1.数据基本信息

def test01():

    # 加载数据集
    train = CIFAR10(root='root', train=True, download=True, transform=Compose([ToTensor()]))
    valid = CIFAR10(root='root', train=False, download=True, transform=Compose([ToTensor()]))

    # 数据集数量
    print('训练集数量', len(train.targets))
    print('测试集数量', len(valid.targets))

    # 数据集的形状
    print('数据集的形状', train[0][0].shape)

    # 数据集类别
    print('数据集类别', train.class_to_idx)

# 2.数据加载器构建
def test02():
    train = CIFAR10(root='root', train=True, transform=Compose([ToTensor()]))
    dataloader = DataLoader(train, batch_size=8, shuffle=True)

    for x, y in dataloader:
        print(x.shape)
        print(y)
        break


# 3.搭建图像分类网络



if __name__ == '__main__':
    test02()