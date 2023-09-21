import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import time
from sklearn.preprocessing import StandardScaler

# 1. 构建数据集
def create_dataset():

    # 读取数据
    data_train = pd.read_csv('data/train.csv')
    # # 将特征值和目标值划分
    x, y = data_train.iloc[:, :-1], data_train.iloc[:, -1]

    # print(y_train)
    # 数据类型转换
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 数据集划分(因为使用的划分好的数据集，所以下面的内容不需要)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)

    # 数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_valid = transfer.transform(x_valid)


    # 构建pytorch对象
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.tensor(y_valid.values))

    # 返回数据
    return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y_train))

train_dataset, valid_dataset, input_dim, class_num = create_dataset()


# 2. 构建分类的网络模型
class PhonePriceModel(nn.Module):

    def __init__(self, input_dim, class_num):
        # 调用父类初始化
        super(PhonePriceModel, self).__init__()

        # 定义网络层
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 128)
        # 输出层
        self.linear5 = nn.Linear(128, class_num)

    def _activation(self, x):
        return torch.sigmoid(x)

    def forward(self, x):

        x = self._activation(self.linear1(x))
        x = self._activation(self.linear2(x))
        x = self._activation(self.linear3(x))
        x = self._activation(self.linear4(x))
        output = self.linear5(x)


        return output

# 3 编写训练函数
def train():

    # 固定随机数种子
    torch.manual_seed(3407)

    # 初始化网络
    model = PhonePriceModel(input_dim, class_num)
    model = model.cuda()
    # 损失函数,会首先对数据进行softmax，再进行交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 训练轮数
    num_epochs = 300

    for epoch_idx in range(num_epochs):

        # 初始化数据加载器
        dataloder = DataLoader(train_dataset, shuffle=True, batch_size=8)

        # 训练时间
        start_time = time.time()
        # 计算损失
        total_loss = 0.0
        total_num = 0.0 
        # 预测正确的样本数量
        corrent = 0

        for x, y in dataloder:

            # 将数据传入网络
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            # 计算损失
            loss = criterion(output, y)
            # 梯度清理
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 累计总样本
            total_num += len(y)
            # 累计总损失
            total_loss += loss.item() * len(y)

            # 预测正确的样本数量
            y_pred = torch.argmax(output, dim=-1)
            corrent += ((y_pred == y).sum().item())

        print('epoch: %4s loss: %.2f time: %.2fs acc: %.2f' % (epoch_idx+1, total_loss/total_num, time.time()-start_time, corrent/total_num) )
    # 模型保存
    torch.save(model.state_dict(), 'model/phone-predict.pth')

# 4. 评估函数
def test():

    # 1. 加载模型
    model = PhonePriceModel(input_dim, class_num)
    model = model.cuda()
    model.load_state_dict(torch.load('model/phone-predict.pth'))
    # 2. 构建测试数据集加载器
    dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    # 3. 计算准确率
    correct = 0.0

    for x, y in dataloader:
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        y_pred = torch.argmax(output, dim=-1)
        correct += ((y_pred == y).sum().item())

    print('acc: %0.5f' % (correct / len(valid_dataset)))

if __name__ == '__main__':
    train()
    test()