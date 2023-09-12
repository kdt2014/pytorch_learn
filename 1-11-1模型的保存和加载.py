import torch
import torch.nn as nn
import pickle

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.linear2 = nn.Linear(input_size * 2, output_size)

    def forward(self, inputs):
        inputs = self.linear1(inputs)
        outputs = self.linear2(inputs)
        return outputs

def test01():
    model = Model(128, 10)
    # model()
    torch.save(model, 'model/test_model_save.pth', pickle_module=pickle, pickle_protocol=2)

def test02():
    model = torch.load('model/test_model_save.pth', pickle_module=pickle, map_location='cpu')
    print(model)


if __name__ == '__main__':
    test02()