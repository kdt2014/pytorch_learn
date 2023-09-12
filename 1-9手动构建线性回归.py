import torch
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import random
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
# 构建数据集
def create_dataset():

    x ,y , coef = make_regression(n_samples=100,
                                  n_features=1,
                                  noise=10,
                                  coef=True,
                                  bias=14.5,
                                  random_state=0)

    # 将构建的数据转换为张量类型

    x = torch.tensor(x)
    y = torch.tensor(y)

    return x, y, coef

# 构建数据加载器

def data_loader(x, y, batch_size):

    # 计算样本数量
    data_len = len(y)
    #构建数据索引
    data_index = list(range(data_len))
    #数据集打乱
    random.shuffle(data_index)
    # 计算总batch数量
    batch_number = data_len // batch_size

    for idx in range(batch_number):

        start = idx * batch_size
        end = start + batch_size

        batch_train_x = x[start:end]
        batch_train_y = y[start:end]

        yield batch_train_x, batch_train_y

def test01():

    x,y = create_dataset()
    plt.scatter(x, y)
    plt.show()

    for x, y in data_loader(x, y, batch_size=10):
        print(y)

# 构建假设函数

w = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
b = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

def linear_regression(x):
    return w * x + b

#损失函数

def square_loss(y_pred, y_true):
    return (y_pred-y_true) ** 2

# 优化方法
def sgd(lr=1e-2):
    # 除以16是使用批次样本的平均梯度值
    w.data = w.data - lr*w.grad.data / 16
    b.data = b.data - lr * b.grad.data / 16

def train():

    # 加载数据集
    x, y, coef = create_dataset()
    # 定义训练相关的参数
    epochs = 100
    learning_rate = 0.01
    # 存储训练过程中的信息
    epoch_loss = []
    total_loss = 0.0
    train_samples = 0

    for _ in range(epochs):

        for train_x, train_y in data_loader(x, y, batch_size=16):

            # 1. 将训练样本送入到模型进行预测
            y_pred = linear_regression(train_x)

            # 2. 计算预测值和真值之间的平方损失
            loss = square_loss(y_pred, train_y.reshape(-1,1)).sum()
            total_loss += loss.item()
            train_samples += len(train_y)

            # 3. 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()

            if b.grad is not None:
                b.grad.data.zero_()

            # 4. 自动微分，反向传播
            loss.backward()

            # 5. 参数更新
            sgd(learning_rate)

            print('loss: %.10f' % (total_loss / train_samples))
        # 记录每一个epoch的平均损失
        epoch_loss.append(total_loss / train_samples)

    # 先绘制数据集散点图
    plt.scatter(x, y)
    #绘制拟合直线
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * w + b for v in x])
    y2 = torch.tensor([v * coef + 14.5 for v in x])

    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()

    #打印损失变化曲线
    plt.plot(range(epochs), epoch_loss)
    plt.grid()
    plt.title('损失变化曲线')
    plt.show()





if __name__ == '__main__':
    train()
