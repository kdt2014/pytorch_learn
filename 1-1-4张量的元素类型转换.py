import torch


# 1 使用type函数转换


def test01():
    data = torch.full([2,3],10)
    print(data.dtype)
    data = data.type(torch.DoubleTensor)
    print(data.dtype)

# 2 使用具体类型函数转换
def test02():
    data = torch.full([2,3],10)
    print(data.dtype)
    # 转换成float64 
    data = data.double()
    print(data.dtype)


if __name__ == '__main__':
    test02()