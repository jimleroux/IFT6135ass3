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
                kernel_size=3, padding=1, stride=2), # 16
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=self.base_channel,
                out_channels=self.base_channel*2,
                kernel_size=3, padding=1, stride=2), # 8
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=self.base_channel*2,
                out_channels=self.base_channel*4,
                kernel_size=3, padding=1, stride=2), # 4
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.base_channel*4,
                out_channels=self.base_channel*8,
                kernel_size=3, padding=1, stride=2), # 2
            nn.ReLU(),
        )
        self.lin_in_dim = self.base_channel*8*2*2
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
            nn.Tanh() # 32
        )

    def forward(self, x):
        return self.network(x.unsqueeze(-1).unsqueeze(-1))


class UpsampleGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=latent_dim, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(), # 4

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(), # 8

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(), # 16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh() # 32
        )

    def forward(self, x):
        return self.network(x.unsqueeze(-1).unsqueeze(-1))
