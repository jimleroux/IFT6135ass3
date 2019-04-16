#!/usr/bin/env python3

import torch
import torch.nn as nn
from gan_model import UpsampleGenerator

class VAE(nn.Module):
    """
    Reference: https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/VAE.ipynb
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(), #16

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(), #8

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(), #4

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(), #2
        )
        self.lin_in_dim = 2*2*512
        self.lin1 = nn.Sequential(
            nn.Linear(self.lin_in_dim, latent_dim),
            nn.ReLU(),
        )
        self.fc11 = nn.Linear(latent_dim, latent_dim)
        self.fc12 = nn.Linear(latent_dim, latent_dim)

        self.decoder = UpsampleGenerator(latent_dim)
    

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        z = self.encoder(x)
        z = z.view(-1, self.lin_in_dim)
        z = self.lin1(z)
        mu = self.fc11(z)
        logvar = self.fc12(z)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat
