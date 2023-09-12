import torch
import numpy as np

def test():
    # 查看张量形状

    torch.manual_seed(0)
    data = torch.randint(0,10,[4,5])

    # 查看张量形状
    print(data.shape,data.shape[0],data.shape[1])
    print(data.size(), data.size(0), data.size(1))

    # 修改张量形状
    new_data = data.reshape(2,10)
    print(new_data)

    # 使用-1省略形状
    new_data = data.reshape(5,-1)
    print(new_data)




if __name__ == '__main__':
    test()