import torch

# 1 标量的梯度计算
# y=x^2=20
def test01():

    # 对于需要求导的张量，requires_grad=True
    x = torch.tensor(10,requires_grad=True, dtype=torch.float64)

    # 对x的中间计算
    f = x ** 2 +20

    # 自动微分
    f.backward()

    # 访问梯度
    print(x.grad)

# 2 向量的梯度计算
def test02():

    # 对于需要求导的张量，requires_grad=True
    x = torch.tensor([10,20,30,40],requires_grad=True, dtype=torch.float64)

    # 对x的中间计算
    y1 = x ** 2 +20
    y2 = y1.mean()

    # 注意，自动微分的时候，必须是一个标量

    # 自动微分
    y2.backward()

    # 访问梯度
    print(x.grad)

# 3 多标量的梯度计算
def test03():
# y = x1**2 + x2**2 +x1*x2
    x1 = torch.tensor(10, requires_grad=True, dtype=torch.float64)
    x2 = torch.tensor(20, requires_grad=True, dtype=torch.float64)
    # 中间计算过程
    y = x1 ** 2 + x2 ** 2 + x1 * x2

    # 自动微分
    y.backward()

    # 访问梯度
    print(x1.grad)
    print(x2.grad)

# 4 多向量的梯度计算
def test04():

    # 对于需要求导的张量，requires_grad=True
    x1 = torch.tensor([10,20],requires_grad=True, dtype=torch.float64)
    x2 = torch.tensor([30, 40], requires_grad=True, dtype=torch.float64)

    # 对x的中间计算
    y = x1**2 + x2**2 + x1*x2

    # 注意，自动微分的时候，必须是一个标量
    y = y.sum()
    # 自动微分
    y.backward()

    # 访问梯度
    print(x1.grad)
    print(x2.grad)


if __name__ == '__main__':
        test04()
