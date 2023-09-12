import torch
import torch.nn as nn
import torch.optim as optim


# 1 损失函数的用法

def test01():
    # 初始化平方数损失函数对象
    criterion = nn.MSELoss()
    # 该类内部重写了 __call__ 方法，所以对象可以当做函数使用
    y_pred = torch.randn(3, 5, requires_grad=True)
    y_true = torch.randn(3, 5)
    # 计算损失
    loss = criterion(y_pred, y_true)

    print(loss)


# 2. 假设函数
def test02():
    # 输入数据的特征必须要有10个
    # 输出数据的特征有5个
    model = nn.Linear(in_features=10, out_features=5)
    # 输入数据
    inputs = torch.randn(4, 10)
    # nn.Linear实现了 __call__ 方法，可以直接当函数使用
    y_pred = model(inputs)
    print(y_pred.shape)

# 3. 优化方法

def test03():
    model = nn.Linear(in_features=10, out_features=5)
    # 优化方法的作用是更新模型参数
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # 在backward()函数调用之前，需要梯度清零
    optimizer.zeros_grad()

    # 此处省略了backward()的调用，假设该函数调用完毕

    # 更新模型参数
    optimizer.step()




if __name__ == '__main__':
    test02()