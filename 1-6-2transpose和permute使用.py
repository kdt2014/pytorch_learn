import torch
import numpy as np


# 1 transpose函数
def test01():
    torch.manual_seed(0)
    data = torch.randint(0,10,[3,4,5])

    new_data = data.reshape(4,3,5)
    print(new_data.shape)

    # 直接交换两个维度的值
    new_data=torch.transpose(data,0,2)
    print(new_data.shape)

    # 缺点：一次只能交换两个维度


# 2 permute函数
def test02():
    torch.manual_seed(0)
    data = torch.randint(0,10,[3,4,5])

    new_data = torch.permute(data,[1,2,0])
    print(new_data.shape)


if __name__ == '__main__':
        test02()