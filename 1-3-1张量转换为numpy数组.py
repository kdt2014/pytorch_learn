import torch

# 1 张量转换为numpy数组

def test01():
    data_tesnor = torch.tensor([2,3,4])
    # 转换为numpy数组
    data_numpy = data_tesnor.numpy()

    print('data_tesnor:',data_tesnor)
    print('data_numpy:',data_numpy)

# 2 张量和numpy数组共享内存

def test02():

        data_tesnor = torch.tensor([2,3,4])
        # 转换为numpy数组
        data_numpy = data_tesnor.numpy()

        print('data_tesnor:',data_tesnor)
        print('data_numpy:',data_numpy)

        # 修改numpy数组
        data_numpy[0] = 100
        print('data_tesnor:',data_tesnor)
        print('data_numpy:',data_numpy)

        # 修改张量
        data_tesnor[1] = 200
        print('data_tesnor:',data_tesnor)
        print('data_numpy:',data_numpy)

# 3 使用copy函数实现不贡献内存
def test03():
    data_tesnor = torch.tensor([2, 3, 4])
    # 转换为numpy数组
    data_numpy = data_tesnor.numpy().copy()


    # 修改numpy数组
    data_numpy[0] = 100
    print('data_tesnor:', data_tesnor)
    print('data_numpy:', data_numpy)

    # 修改张量
    data_tesnor[1] = 200
    print('data_tesnor:', data_tesnor)
    print('data_numpy:', data_numpy)

if __name__ == '__main__':
    test03()