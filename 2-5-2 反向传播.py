import torch
import torch.nn as nn
import torch.optim as optim

# 1. 搭建网络
# 自己编写的网络类需要继承父类 nn.Module
class Net(nn.Module):

    def __init__(self):
        # 注意：必须手动调用父类函数
        super(Net, self).__init__()

        self.linear1 = nn.Linear(in_features=2, out_features=2)
        self.linear2 = nn.Linear(in_features=2, out_features=2)

        # 手动对参数进行初始化

        self.linear1.weight.data = torch.tensor([[0.15, 0.20], [0.25, 0.30]])
        self.linear2.weight.data = torch.tensor([[0.40, 0.45], [0.50, 0.55]])
        self.linear1.bias.data = torch.tensor([[0.35, 0.35]])
        self.linear2.bias.data = torch.tensor([[0.60, 0.60]])

        # 前向传播
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        # 正向传播结束后需要返回输出结果
        return x

if __name__ == '__main__':
    # 输入数据，注意：二维列表表示批次样本的输入
    inputs = torch.tensor([[0.05, 0.10]])
    # 真实值
    target = torch.tensor([[0.01, 0.99]])
    # 初始化网络对象
    net = Net()
    output = net(inputs)
    # print(output)

    # 计算误差
    loss = 1/2 * torch.sum((output-target)**2)
    print(loss)

    # 反向传播
    # 构建优化器
    optimizer = optim.SGD(net.parameters(), lr=0.5)
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 参数更新
    optimizer.step()
    # 打印参数
    print(net.linear1.weight.grad.data)
    print(net.linear2.weight.grad.data)

    # 打印更新之后的参数
    print(net.linear1.weight.data)
    print(net.linear2.weight.data)