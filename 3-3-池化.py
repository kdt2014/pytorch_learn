import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1.api基本使用
def test01():

    inputs = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).float()  # 维度 3*3
    # 池化输入： B * C * H * W
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print(inputs.shape)

    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=2, stride=1,  padding=0)
    output = polling(inputs)
    print(output.shape)

    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=1,  padding=0)
    output = polling(inputs)
    print(output.shape)

# 2.stride步长
def test02():

    inputs = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 7], [6, 7, 8, 9], [10, 11, 12, 13]]).float()  # 维度 4*4
    # 池化输入： B * C * H * W
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print(inputs.shape)

    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=2, stride=2,  padding=0)
    output = polling(inputs)
    print(output.shape)

    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=2,  padding=0)
    output = polling(inputs)
    print(output.shape)

# 3.padding
def test03():

    inputs = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).float() # 维度 3*3
    # 池化输入： B * C * H * W
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print(inputs.shape)

    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=2, stride=1,  padding=1)
    output = polling(inputs)
    print(output.shape)

    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=1,  padding=1)
    output = polling(inputs)
    print(output.shape)

# 4.多通道池化
def test04():

    inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                           [[4, 6, 9], [10, 4, 5], [11, 7, 12]],
                           [[6, 7, 2], [377, 47, 54], [63, 72, 81]]
                           ]).float()  # 维度 3*3*3
    # 池化输入： B * C * H * W
    inputs = inputs.unsqueeze(0)
    print(inputs.shape)

    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=2, stride=1,  padding=0)
    output = polling(inputs)
    print(output.shape)

    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2, stride=1,  padding=0)
    output = polling(inputs)
    print(output.shape)

    # 注意：池化只会改变特征图的大小，不会改变输入图像的通道数量


# 4.多通道池化
def test05():
    # 读取图像：512*510*3  ----> (H, W, C)
    img = plt.imread('data/lena.png')
    # 池化输入： B * C * H * W
    inputs = torch.tensor(img).permute(2, 0, 1)   # 3*512*510
    inputs = inputs.unsqueeze(0)  # 1*3*512*510
    print(inputs.shape)

    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
    output = polling(inputs)
    print(output.shape)

    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=3, stride=3, padding=0)
    output = polling(inputs)
    print(output.shape)

    # 注意：池化只会改变特征图的大小，不会改变输入图像的通道数量

if __name__ == '__main__':
    test05()