import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# 确保 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(torch.cuda.is_available())
print(device)

# 1.网络模型
class ImageClassification(nn.Module):

    def __init__(self):
        # 调用父类函数初始化
        super(ImageClassification, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 128, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义线性层
        self.linear1 = nn.Linear(128*6*6, 12048)
        self.linear2 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, 10)

    # 定义前向传播
    def forward(self, x):
        # 第一个卷积池化
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # 第二个卷积池化
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # 将特征图送入到全连接层
        x = x.reshape(x.size(0), -1)

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # out = self.out(x)

        return self.out(x)

# 2.训练函数

def train():

    # 加载数据集
    cifar10 = CIFAR10('data', train=True, transform=Compose([ToTensor()]))
    # 初始化网络
    model = ImageClassification()
    # 将模型移动到 GPU
    model.to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 训练轮次
    epochs = 100

    for epoch_idx in range(epochs):

        # 构建数据加载器
        dataloader = DataLoader(cifar10, batch_size=32, shuffle=True)
        # 样本数量
        sum_num = 0
        # 损失总和
        total_loss = 0.0
        # 开始时间
        start_time = time.time()
        # 正确样本数量
        correct = 0

        # 开始训练
        for x, y in dataloader:
            # 将数据移动到GPU
            x, y = x.to(device), y.to(device)
            # 传入数据
            output = model(x)
            # 计算损失
            loss = criterion(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 统计信息
            correct += (torch.argmax(output, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            sum_num += len(y)
        print('epoch: %.2s loss: %.5f acc: %.2f time:%.2fs' % (epoch_idx+1, total_loss/sum_num, correct/sum_num, time.time()-start_time))

    torch.save(model.state_dict(), 'model/image_classification_model.pth')

# 编写预测函数
def test():
    # 加载数据集
    cifar10 = CIFAR10(root='data', train=False, download=True, transform=Compose([ToTensor()]))
    # 构建数据加载器
    dataloader = DataLoader(cifar10, batch_size=32, shuffle=False)
    # 加载模型
    model = ImageClassification()
    model.load_state_dict(torch.load('model/image_classification_model.pth'))
    # 模型有两种状态： 训练状态、预测状态(模式)
    model.eval()

    # 统计信息
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        # 数据传入网络
        output = model(x)
        # 统计正确个数
        total_correct += (torch.argmax(output, dim=-1) == y).sum()
        total_samples += len(y)

    # 打印准确率
    print('acc: %.2f' %(total_correct/total_samples))




if __name__ == '__main__':
    test()


