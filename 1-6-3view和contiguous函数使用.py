import torch
import numpy as np


# 1 view函数的使用
def test01():

    data = torch.tensor([[10,20,30],[40,50,60]])
    data = data.view(3,2)
    print(data.shape)

    # is_contiguous函数判断张量是否连续
    print(data.is_contiguous())

# 2 view函数使用注意

def test02():
    # 当张量经过transpose和permute之后，函数空间基本不连续
    # 需要先把空间连续，才能使用view函数

    data = torch.tensor([[10,20,30],[40,50,60]])
    print('是否连续',data.is_contiguous())
    data = torch.transpose(data,1,0)
    print('是否连续', data.is_contiguous())

    # 此时，在不连续的情况下使用view会怎么样？
    data = data.contiguous().view(2,3)
    print(data)

if __name__ == '__main__':
        test02()