import torch
import numpy as np

# 1 from_numpy 方法

def test01():
    data_numpy = np.array([2,3,4])
    data_tensor = torch.from_numpy(data_numpy.copy())
    print(type(data_tensor))
    print(type(data_numpy))

    # 默认共享内存
    data_numpy[0] = 10
    print('data_tesnor:', data_tensor)
    print('data_numpy:', data_numpy)


# 2 torch.tensor()方法

def test02():
    data_numpy = np.array([2, 3, 4])
    data_tensor = torch.tensor(data_numpy)

    data_numpy[0] = 10
    print('data_tesnor:', data_tensor)
    print('data_numpy:', data_numpy)



if __name__ == '__main__':
    test02()