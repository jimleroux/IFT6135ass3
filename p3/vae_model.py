#!/usr/bin/env python3

import torch
import torch.nn as nn
from gan_model import UpsampleGenerator, ConvBlock

class VAE(nn.Module):
    """
    Reference: https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/VAE.ipynb
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        self.encoder = ConvBlock(base_channel)
        self.lin_in_dim = 2*2*base_channel*8
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
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar 
