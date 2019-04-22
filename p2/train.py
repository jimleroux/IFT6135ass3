import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np

from dataloader import TrainDataset, ValidDataset, TestDataset
from VAE import VAE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):
    trainset = TrainDataset()
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    vae = VAE().to(DEVICE)
    vae.fit(trainloader, n_epochs=args.num_epochs, lr=args.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30)
    parser.add_argument('--batch_size',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        type=float,
                        default=0.0006)
    args = parser.parse_args()
    train(args)


