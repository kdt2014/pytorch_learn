import torch


# 1 使用@运算符

def test01():

    # 形状：3行2列
    data1 = torch.tensor([[1,2],[3,4],[5,6]])
    # 形状：2行2列
    data2 = torch.tensor([[5, 6], [7, 8]])
    data = data1 @ data2
    print(data)

# 2 使用mm函数

def test02():

    # 要求输入的张量必须是二维的
    # 形状：3行2列
    data1 = torch.tensor([[1,2],[3,4],[5,6]])
    # 形状：2行2列
    data2 = torch.tensor([[5, 6], [7, 8]])
    data = torch.mm(data1,data2)
    print(data)

# 3 使用bmm函数

def test03():

        # 第一个维度：表示有几个矩阵（批次）
        # 第二个维度：多少行
        # 第三个维度：多少列

        data1 = torch.randn(3,4,5)
        data2 = torch.randn(3,5,8)
        data = torch.bmm(data1,data2)
        print(data)
        print(data.shape)

# 4 使用matmul函数

def test04():

        # 4.1 二维张量
        data1 = torch.randn(4,5)
        data2 = torch.randn(5,8)
        data = torch.matmul(data1,data2)
        print(data.shape)

        # 4.2 三维张量
        data1 = torch.randn(3,4,5)
        data2 = torch.randn(3,5,8)
        data = torch.matmul(data1,data2)
        print(data.shape)

        # 4.3 三维张量和二维张量
        data1 = torch.randn(3,4,5)
        data2 = torch.randn(5,8)
        data = torch.matmul(data1,data2)
        print(data.shape)

        # 4.4 二维张量和三维张量
        data1 = torch.randn(4,5)
        data2 = torch.randn(3,5,8)
        data = torch.matmul(data1,data2)
        print(data.shape)

        # 4.5 三维张量和一维张量
        data1 = torch.randn(3,4,5)
        data2 = torch.randn(5)
        data = torch.matmul(data1,data2)
        print(data.shape)

        # 4.6 一维张量和三维张量
        data1 = torch.randn(5)
        data2 = torch.randn(3,5,8)
        data = torch.matmul(data1,data2)
        print(data.shape)

        # 4.7 一维张量和二维张量
        data1 = torch.randn(5)
        data2 = torch.randn(5,8)
        data = torch.matmul(data1,data2)
        print(data.shape)

        # 4.8 二维张量和一维张量
        data1 = torch.randn(5,8)
        data2 = torch.randn(8)
        data = torch.matmul(data1,data2)
        print(data.shape)








if __name__ == '__main__':
    test04()