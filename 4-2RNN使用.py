import torch
import torch.nn as nn

# 1 一次输入一个词
def test01():
    # 构造网络
    # input_size 输入句子每个词的向量维度128，比如'我'经过词嵌入之后得到了一个128维的表示；
    # hidden_size 隐藏层的大小，隐藏层神经元的个数，影响最终输出数据的维度
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 初始化输入数据
    # 注意输入数据有两个：上一个时间步的隐藏状态，当前时间步的输入
    # 输入数据的形状 (seq_len, batch_size, input_size)
    # 第一个数字：表示句子长度，一个句子由几个词组成，长度不够补零，过长就截断
    # 第二个数字：批量个数，即一次输入几个句子
    # 第三个数字：表示数据维度
    inputs = torch.randn(1, 1, 128)
    # 隐藏层形状 (num_layers, batch_size, hidden_size)
    hn = torch.zeros(1, 1, 256)

    # 将数据送入到循环网络层
    # 输出: output, hn
    output, hn = rnn(inputs, hn)

    print(hn.shape)
    print(output.shape)

# 2 输入句子
def test02():
    # 初始化循环网络层
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 构造输入数据
    inputs = torch.randn(8, 1, 128)
    # 隐藏层形状 (num_layers, batch_size, hidden_size)
    hn = torch.zeros(1, 1, 256)

    # 将数据送入到循环网络层
    # 输出: output, hn
    output, hn = rnn(inputs, hn)

    print(output.shape)
    print(hn.shape)

# 3 RNN输入批次数据
def test03():
    # 初始化循环网络层
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 构造输入数据
    # 输入数据的形状 (seq_len, batch_size, input_size)
    inputs = torch.randn(8, 16, 128)
    # 隐藏层形状 (num_layers, batch_size, hidden_size)
    hn = torch.zeros(1, 16, 256)

    # 将数据送入到循环网络层
    # 输出: output, hn
    output, hn = rnn(inputs, hn)

    print(output.shape)
    print(hn.shape)



if __name__ == '__main__':
    test03()
