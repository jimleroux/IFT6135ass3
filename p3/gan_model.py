#!/usr/bin/env python3

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_channel = 64
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.base_channel,
                kernel_size=7), # 26
            nn.LeakyReLU(),
            
            nn.Conv2d(
                in_channels=self.base_channel,
                out_channels=self.base_channel*2,
                kernel_size=3, stride=2), # 12
            nn.LeakyReLU(),
            
            nn.Conv2d(
                in_channels=self.base_channel*2,
                out_channels=self.base_channel*4,
                kernel_size=3), # 10
            nn.LeakyReLU(),
        )
        self.lin_in_dim = self.base_channel*4*10*10
        self.linear = nn.Linear(self.lin_in_dim, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.lin_in_dim)
        out = self.linear(out)
        return out
    
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=1024, kernel_size=4),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(), # 4

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(), # 8

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(), # 16

            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.LeakyReLU(), # 32
        
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x.unsqueeze(-1).unsqueeze(-1))
