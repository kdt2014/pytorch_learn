import torch
import numpy as np

# 1.根据已有数据创建张量

def test01():

    # 1.1 创建标量
    data = torch.tensor(10)
    print(data)

    # 1。2使用Numpy数组创建张量
    data = np.random.randn(2,3)
    data = torch.tensor(data)
    print(data)
    # 1.3 使用list列表创建张量

    data = [[3.,4.,5.],[6.,7.,8.]]
    data = torch.tensor(data)
    print(data)

# 2.创建指定形状的张量
def test02():

    # 2.1 创建两行三列的张量
    data = torch.Tensor(2,3)
    print(data)

    # 2.2 创建指定值的张量
    # 传递列表
    data = torch.Tensor([2,3])
    print(data)

#3.创建指定类型的张量
# 默认类型是float32
#前面创建的都是默认类型的张量

def test03():
    data = torch.IntTensor(2,3)
    print(data)

    # 如果数据不匹配会发生类型转换，可能导致精度丢失
    data = torch.IntTensor([2.5, 3.5])
    print(data)

if __name__ == '__main__':
    test03()