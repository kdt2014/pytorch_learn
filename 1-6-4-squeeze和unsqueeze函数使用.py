import torch
import numpy as np

# 1 squeeze函数使用

def test01():
    data = torch.randint(0,10,[1,3,1,5])
    print(data.shape)

    # 维度压缩
    new_data = data.squeeze()
    print(new_data.shape)
    new_data = data.squeeze(2)
    print(new_data.shape)


# 2 unsqueeze函数使用
def test02():
    data = torch.randint(0,10,[3,5])
    print(data.shape)

    # 可以指定位置增加维度
    # -1 代表最后一个位置
    new_data = data.unsqueeze(1)
    print(new_data.shape)


if __name__ == '__main__':
        test02()