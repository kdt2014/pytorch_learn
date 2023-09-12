import torch


# 1 均值
def test01():
    torch.manual_seed(0)
    # data = torch.randint(0,10,[2,3 ],dtype=torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()

    print(data.dtype)
    print(data)
    print(data.mean())
    print(data.mean(dim=0))
    print(data.mean(dim=1))

# 2 求和

def test02():
    torch.manual_seed(0)
    # data = torch.randint(0,10,[2,3 ],dtype=torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()

    print(data.dtype)
    print(data)
    print(data.sum())
    print(data.sum(dim=0))
    print(data.sum(dim=1))

# 3 平方
def test03():
    torch.manual_seed(0)
    # data = torch.randint(0,10,[2,3 ],dtype=torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()

    print(data.dtype)
    print(data)
    print(data.pow(2))

# 4 平方根
def test04():
    torch.manual_seed(0)
    # data = torch.randint(0,10,[2,3 ],dtype=torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()

    print(data.dtype)
    print(data)
    print(data.sqrt())

# 5 e的多少次方
def test05():
    torch.manual_seed(0)
    # data = torch.randint(0,10,[2,3 ],dtype=torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()

    print(data.dtype)
    print(data)
    print(data.exp())

# 6 对数
def test06():
    torch.manual_seed(0)
    # data = torch.randint(0,10,[2,3 ],dtype=torch.float64)
    data = torch.randint(0, 10, [2, 3]).double()

    print(data.dtype)
    print(data)
    print(data.log())  #默认以e为底
    print(data.log2())
    print(data.log10())

if __name__ == '__main__':
        test06()