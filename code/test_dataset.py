import numpy as np
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, df_test):
        self.df_test = df_test

    def __len__(self):
        return len(self.df_test)

    def __getitem__(self, idx):
        return self.df_test.iloc[idx, :].values.astype(np.float32)
