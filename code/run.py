import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from code.train_dataset import TrainingDataset


def main():
    torch.manual_seed(0)

    BATCH_SIZE = 4

    # 1. DatasetとDataLoader
    train = pd.read_csv('../input/train.csv', nrows=10)
    X = train.iloc[:, 1:].values
    y = train.iloc[:, 0].values

    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          y,
                                                          train_size=0.8,
                                                          random_state=0)
    train_dataset = TrainingDataset(X_train, y_train)
    valid_dataset = TrainingDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == '__main__':
    main()
