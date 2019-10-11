import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrainingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), self.y[idx]


def main():
    data_num = 10
    BATCH_SIZE = 4

    X_train = torch.ones(size=(data_num, 784))
    y_train = torch.ones(size=(data_num,))

    train_dataset = TrainingDataset(X=X_train, y=y_train)

    assert data_num == len(train_dataset)

    x, y = next(iter(train_dataset))
    assert torch.Size([784]) == x.size()
    assert 1.0 == y

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for images, labels in train_loader:
        print(images.size(), labels.size())


if __name__ == '__main__':
    main()
