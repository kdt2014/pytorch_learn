import torch

# 1. 不修改原数据的计算
def test01():
    # 第一个值：开始值
    # 第二个值：结束值
    # 第三个值：形状
    data = torch.randint(0,10,[2,3])
    print(data)

    # 计算完成后，会返回一个新的张量，不会修改原来的张量
    data = data.add(10)
    print(data)

    # data.sub() # 减法
    # data.mul() # 乘法
    # data.div() # 除法
    # data.neg() # 取反

    # 2. 修改原数据的计算
def test02():
    # 第一个值：开始值
    # 第二个值：结束值
    # 第三个值：形状
    data = torch.randint(0,10,[2,3])
    print(data)

    # 计算完成后，会返回一个新的张量，不会修改原来的张量
    data.add_(10)
    print(data)

    # data.sub_() # 减法
    # data.mul_() # 乘法
    # data.div_() # 除法
    # data.neg_() # 取反

if __name__ == '__main__':
    test02()