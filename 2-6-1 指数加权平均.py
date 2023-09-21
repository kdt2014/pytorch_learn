import torch
import matplotlib.pyplot as plt
# 1. 没有指数加权平均
def test01():

    # 固定随机种子
    torch.manual_seed(0)

    # 产生随机30天的温度
    temperature = torch.randn(size=[30,]) * 10

    # 绘制温度变化曲线平均温度值
    days = torch.arange(1, 31, 1)
    plt.plot(days, temperature)
    plt.show()



# 2. 有指数加权平均
def test02(beta=0.9):
    # 固定随机种子
    torch.manual_seed(0)
    # 产生随机30天的温度
    temperature = torch.randn(size=[30,]) * 10
    # days = torch.arange(1, 31, 1)

    # 存储历史指数加权平均值
    exp_weight_avg = []

    for idx, temp in enumerate(temperature, 0):

        if idx == 0:
            exp_weight_avg.append(temp)
            continue
        new_temp = exp_weight_avg[idx-1] * beta + (1-beta)*temp
        exp_weight_avg.append(new_temp)

    # 绘制温度变化曲线指数加权平均值
    days = torch.arange(1, 31, 1)
    plt.plot(days, exp_weight_avg, 'o-r')
    plt.show()

if __name__ == '__main__':
    test02(0.5)