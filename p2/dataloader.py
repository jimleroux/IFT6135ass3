import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

PATH_DIR = "./dataset/numpy_data/"

class TrainDataset(Dataset):
    def __init__(self):
        self.data = np.load(PATH_DIR + "train.npy")

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.reshape(image, (1, 28, 28))
        image = torch.from_numpy(image)
        return image

class ValidDataset(Dataset):
    def __init__(self):
        self.data = np.load(PATH_DIR + "valid.npy")

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.reshape(image, (1, 28, 28))
        image = torch.from_numpy(image)
        return image

class TestDataset(Dataset):
    def __init__(self):
        self.data = np.load(PATH_DIR + "test.npy")

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.reshape(image, (1, 28, 28))
        image = torch.from_numpy(image)
        return image