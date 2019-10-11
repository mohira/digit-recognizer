import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader

from code.my_simple_net import MySimpleNet
from code.test_dataset import TestDataset
from code.train_dataset import TrainingDataset


def main():
    torch.manual_seed(0)

    BATCH_SIZE = 16

    # 1. DatasetとDataLoader
    train = pd.read_csv('../input/train.csv', nrows=160)
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

    # 2. モデル(ネットワーク)
    model: nn.Module = MySimpleNet()

    # 最適化アルゴリズムと損失関数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 3. 学習
    NUM_EPOCHS = 2

    model.train()  # 学習モード

    from tqdm import tqdm
    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        print(f'Epoch: {epoch}')

        for images, labels in train_loader:
            # 勾配初期化
            optimizer.zero_grad()

            # 順伝播計算
            outputs = model(images)

            # 損失の計算
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()

            # 重みの更新
            optimizer.step()

            print(loss.item())

    # 4. TestDataでの予測
    df_test = pd.read_csv('../input/test.csv')

    test_dataset = TestDataset(df_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()

    predictions = np.array([], dtype=np.int)
    with torch.no_grad():
        for images in test_loader:
            outputs = model(images)

            _, y_pred = torch.max(outputs, dim=1)
            y_pred_label = y_pred.numpy()

            predictions = np.append(predictions, y_pred_label)

    print(predictions)
    print(predictions.shape)

    submit_data = pd.read_csv('../input/sample_submission.csv')
    submit_data['Label'] = predictions

    submit_data.to_csv('simple_nn.csv', index=False)


if __name__ == '__main__':
    main()
