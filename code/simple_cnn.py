import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(3, 6, kernel_size=(2, 2))

        self.fc1 = nn.Linear(6 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 10)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = x.view(-1, 6 * 6 * 6)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


def main():
    images = torch.ones(size=(5, 1, 28, 28))  # (N, C, W, H)

    net = SimpleCNN()
    outputs = net(images)

    assert torch.Size([5, 10]) == outputs.size()


if __name__ == '__main__':
    main()
