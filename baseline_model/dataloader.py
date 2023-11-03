import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import torch.nn.functional as F


class FeaturesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_idx = {label: idx for idx, label in enumerate(sorted(set(self.data["label"])))}
        self.mean = torch.tensor(self.data.iloc[:,2:-1].mean(), dtype=torch.float32) # Mean for each column
        self.std = torch.tensor(self.data.iloc[:,2:-1].std(), dtype=torch.float32) # Standard variation for each column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = torch.tensor(self.data.iloc[idx, 2:-1], dtype=torch.float32)
        label = torch.tensor(self.label_idx[self.data.iloc[idx, -1]])
        
        input = (input - self.mean)/self.std

        if self.transform:
            input = self.transform(input)

        return input, label


def get_dataloader(path, batch_size=1, transform=None):
    trainset = FeaturesDataset(csv_file=path, transform=transform)

    # Split into train and validation
    indices = list(range(len(trainset)))
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(indices)
    split_train = int(len(indices) * 0.7) # split at 70%
    split_test = int(len(indices) * 0.85) # split at 85%

    # Finally, train is 70% of total data, val: 15% and test 15%

    # split into training and validation indices
    train_indices, val_indices, test_indices = indices[:split_train], indices[split_train:split_test], indices[split_test:]
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=1, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=1, sampler=val_sampler)
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=1, sampler=test_sampler)

    return train_loader, val_loader, test_loader