import torch


def test():
    t1 = torch.tensor(30)
    t2 = torch.tensor([30])
    t3 = torch.tensor([[30]])

    print(t1.shape)
    print(t2.shape)
    print(t3.shape)




if __name__ == '__main__':
    test()