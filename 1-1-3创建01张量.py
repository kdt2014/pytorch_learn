import torch


#1 创建全为0的张量

def test01():
    # 1.1创建全为0的张量
    data = torch.zeros(2,3)
    print(data)

    #1.2 根据其他张量的形状，创建全0张量
    data = torch.zeros_like(data)
    print(data)


#2 创建全为1的张量
def test02():
    # 1.1创建全为1的张量
    data = torch.ones(2,3)
    print(data)

    #1.2 根据其他张量的形状，创建全0张量
    data = torch.ones_like(data)
    print(data)

#3 创建指定值的张量

def test03():

    #3.1 创建2行3列，值全部为10的张量
    data = torch.full([2,3],10)
    print(data)

    # 3.2 创建一个形状和data一样，但是值全部是100的张量
    data = torch.full_like(data, 20)
    print(data)
if __name__ == '__main__':
    test03()