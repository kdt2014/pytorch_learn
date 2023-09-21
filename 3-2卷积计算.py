import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 图像显示函数
def show(img):
    # 要求输入图像是H*W*C
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# 1. 单个卷积核
def test01():
    # 读取图像：512*510*3  ----> (H, W, C)
    img = plt.imread('data/lena.png')
    print(img.shape)
    show(img)

    # 构建卷积核
    # in_channels 输入图像通道数
    # out_channels 指输入一个图像后，产生几个特征图，也就是卷积核的数量
    # kernel_size 表示卷积核的大小
    # stride 表示步长
    # padding 表示填充
    conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

    # 额外注意：卷积层对输入图像的数据有形状要求，(BatchSize, Channel, Height, Width)
    # 将(H, W, C) ---> (C, H, W)
    img = torch.tensor(img).permute(2, 0, 1)
    print(img.shape)

    # 将(C, H, W)---> (B, C, H, W)
    new_img = img.unsqueeze(0)
    print(new_img.shape)

    # 将数据送入卷积层计算
    new_img = conv(new_img)
    print(new_img.shape)

    # 将特征图数据转换成H*W*C
    new_img = new_img.squeeze(0)
    new_img = new_img.permute(1, 2, 0)

    # 显示特征图
    show(new_img.detach().numpy())


# 2. 多个卷积核
def test02():
    # 读取图像：512*510*3  ----> (H, W, C)
    img = plt.imread('data/lena.png')
    print(img.shape)
    show(img)

    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 额外注意：卷积层对输入图像的数据有形状要求，(BatchSize, Channel, Height, Width)
    # 将(H, W, C) ---> (C, H, W)
    img = torch.tensor(img).permute(2, 0, 1)
    print(img.shape)

    # 将(C, H, W)---> (B, C, H, W)
    new_img = img.unsqueeze(0)
    # print(new_img.shape)

    # 将数据送入卷积层计算
    new_img = conv(new_img)
    print(new_img.shape)

    # 将特征图数据转换成H*W*C
    new_img = new_img.squeeze(0)
    new_img = new_img.permute(1, 2, 0)
    print(new_img.shape)

    # 显示特征图
    show(new_img[:, :, 0].detach().numpy())
    show(new_img[:, :, 1].detach().numpy())
    show(new_img[:, :, 2].detach().numpy())
    # print(new_img[:, :, 0].shape)


if __name__ == '__main__':
    test02()