import torch


# 1 使用cuda函数指定运算设备

def test01():

    data = torch.tensor([10,20,30])
    print('data的设备：',data.device)

    # 将数据放到GPU上
    data = data.cuda()
    print('data的设备：',data.device)

    # 将数据放到CPU上
    data = data.cpu()
    print('data的设备：',data.device)

# 2 直接将数据放到GPU上

def test02():

        # 将数据放到GPU上
        data = torch.tensor([10,20,30],device='cuda')
        print('data的设备：',data.device)

        # 将数据放到CPU上
        data = data.cpu()
        print('data的设备：',data.device)

# 3 使用to函数指定运算设备

def test03():

        # 将数据放到GPU上
        data = torch.tensor([10,20,30])
        print('data的设备：',data.device)

        # 将数据放到GPU上
        data = data.to('cuda')
        print('data的设备：',data.device)

        # 将数据放到CPU上
        data = data.to('cpu')
        print('data的设备：',data.device)

# 4 注意：张量和张量之间的运算，要求张量的类型和设备都要一致






if __name__ == '__main__':
    test03()