import torch

# 1.演示错误

def test01():
    x = torch.tensor([10,20], requires_grad=True, dtype=torch.float64)

    # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # print(x.numpy())
    # 正确做法
    print(x.detach().numpy())

# 2.共享数据
def test02():
    # x是叶子结点
    x1 = torch.tensor([10,20], requires_grad=True, dtype=torch.float64)
    # detach()分离出新的张量
    x2 = x1.detach()

    print(id(x1.data), id(x2.data))
    x2[0] = 100
    print(x1)
    print(x2)
    # 通过结果发现，x2张量不存在requires_grad=True
    # 表示：对x1的任何计算都会影响到梯度计算
    # 但是，对x2的任何计算不会影响到x1的梯度计算
    print(x1.requires_grad)
    print(x2.requires_grad)


if __name__ == '__main__':
        test02()