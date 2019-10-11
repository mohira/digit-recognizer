import torch
from torch import nn


class MySimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def main():
    model = MySimpleNet()

    data_num = 5
    dummy_inputs = torch.ones(size=(data_num, 784))

    outputs = model(dummy_inputs)

    assert torch.Size([data_num, 10]) == outputs.size()


if __name__ == '__main__':
    main()
