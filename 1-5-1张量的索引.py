import torch


# 1 简单的行列索引

def test01():
    torch.manual_seed(0)
    data = torch.randint(0,10,[4,5])
    print(data)
    print('-' * 30)

    # 1.1 获得指定行
    print(data[2])

    # 1.2 获得指定列
    print(data[:,2])

    print(data[1,2])
    print(data[:3, 2])

# 2 列表索引

def test02():
    torch.manual_seed(0)
    data = torch.randint(0,10,[4,5])
    print(data)
    print('-' * 30)

    # 如果索引行列都是1维列表，那么两个列表的长度必须相等
    # 表示获得（0,0），（2,1），（3,2）的元素
    print(data[[0,2,3],[0,1,2]])


    # 表示获得第0，2,3行的0，1,3列
    print(data[[[0],[2],[3]],[0,1,3]])

if __name__ == '__main__':
    test02()