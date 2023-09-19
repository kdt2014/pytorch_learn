import torch

if __name__ == '__main__':

    x = torch.tensor(1.0)
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    f = (torch.exp(-(w*x+b))+1)**-1

    f.backward()

    print(w.grad)

