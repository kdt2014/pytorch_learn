import torch
import torch.nn as nn

# 1. 创建和使用dropout

def test01():

    # 初始化dropout对象,每个元素有0.8的概率丢失变为0
    dropout = nn.Dropout(p=0.8)
    # 初始化数据
    inputs = torch.randint(0, 10, size=[5, 8]).float()
    print(inputs)
    print('-' * 50)

    # 将inputs数据经过dropout
    outputs = dropout(inputs)

    print(outputs)

# 2. dropout随机丢弃对网络参数的影响
def test02():
    torch.manual_seed(0)
    # 初始化权重
    w = torch.randn(15,1,requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0,10, size=[5, 15]).float()

    # 计算梯度
    y = x @ w
    y = y.sum()
    y.backward()
    print('梯度:', w.grad.squeeze().numpy())

def test03():
    torch.manual_seed(0)
    # 初始化权重
    w = torch.randn(15,1,requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0,10, size=[5, 15]).float()
    # dropout随机丢弃
    dropout = nn.Dropout(p=0.8)
    x = dropout(x)
    # 计算梯度
    y = x @ w
    y = y.sum()
    y.backward()
    print('梯度:', w.grad.squeeze().numpy())



if __name__ == '__main__':
    test02()
    print('-' * 50)
    test03()