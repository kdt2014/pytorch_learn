import torch

# 1.控制梯度计算
def test01():
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64)
    print(x.requires_grad)

    # 1.第一中方法
    with torch.no_grad():
        y = x**2
    print(y.requires_grad)

    # 2 针对函数的
    @torch.no_grad()
    def my_func(x):
        return x**2
    y = my_func(x)
    print(y.requires_grad)

    # 3 第三种全局方式
    torch.set_grad_enabled(False)
    y = x**2
    print(y.requires_grad)

# 2.累计梯度和梯度清零
def test02():
    x = torch.tensor([10,20,30,40], requires_grad=True, dtype=torch.float64)

    # 当我们重复x进行梯度计算的时候，是会将历史的梯度值累加到x.grad 属性当中
    # 希望不要累加历史梯度
    for _ in range(3):
        # 对x的计算过程
        f1 = x**2 + 20
        print(f1)
        # 将向量转换为标量
        f2 = f1.mean()

        # 梯度清零
        if x.grad is not None:
            x.grad.data.zero_()

        # 自动微分
        f2.backward()

        print(x.grad)

# 3. 案例--梯度下降优化
def test03():
    # y = x**2
    # 当x是什么值时，y最小
    # 初始化
    x = torch.tensor(10, requires_grad=True, dtype=torch.float64)
    for _ in range(5000):
        # 正向计算
        y = x**2
        if x.grad is not None:
            x.grad.data.zero_()

        # 自动微分
        y.backward()

        # 更新参数
        x.data = x.data - 0.001 * x.grad

        # 打印x值
        print('%.10f' % x.data)


if __name__ == '__main__':
        test03()
