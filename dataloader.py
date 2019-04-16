import os

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

PATH_DIR = "./dataset/numpy_data/"

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])

class TrainDataset(Dataset):
    def __init__(self, transform=DEFAULT_TRANSFORM):
        self.data = np.load(PATH_DIR + "train.npy")
        self.transform = transform

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.reshape(image, (28, 28))
        image = self.transform(image)
        return image

class ValidDataset(Dataset):
    def __init__(self, transform=DEFAULT_TRANSFORM):
        self.data = np.load(PATH_DIR + "valid.npy")
        self.transform = transform

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.reshape(image, (28, 28))
        image = self.transform(image)
        return image

class TestDataset(Dataset):
    def __init__(self, transform=DEFAULT_TRANSFORM):
        self.data = np.load(PATH_DIR + "test.npy")
        self.transform = transform

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.reshape(image, (28, 28))
        image = self.transform(image)
        return image
