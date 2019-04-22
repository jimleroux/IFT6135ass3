#!/usr/bin/env python3

import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ConvBlock(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        self.base_channel = base_channel
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.base_channel,
                kernel_size=4, padding=1, stride=2), # 16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(
                in_channels=self.base_channel,
                out_channels=self.base_channel*2,
                kernel_size=4, padding=1, stride=2), # 8
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.base_channel*2,
                out_channels=self.base_channel*4,
                kernel_size=4, padding=1, stride=2), # 4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=self.base_channel*4,
                out_channels=self.base_channel*8,
                kernel_size=4, padding=1, stride=2), # 2
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_channel = 64
        self.conv = ConvBlock(self.base_channel)
        self.lin_in_dim = self.base_channel*8*2*2
        self.linear = nn.Linear(self.lin_in_dim, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.lin_in_dim)
        out = self.linear(out)
        return out
    
    def load_conv(self, state_dict):
        self.conv.load_state_dict(state_dict)
    
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=base_channel*8,
                bias=False,
                kernel_size=4),
            nn.BatchNorm2d(num_features=base_channel*8),
            nn.ReLU(True), # 4

            nn.ConvTranspose2d(
                in_channels=base_channel*8,
                out_channels=base_channel*4,
                bias=False,
                kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=base_channel*4),
            nn.ReLU(True), # 8

            nn.ConvTranspose2d(
                in_channels=base_channel*4,
                out_channels=base_channel*2,
                bias=False,
                kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=base_channel*2),
            nn.ReLU(True), # 16

            nn.ConvTranspose2d(
                in_channels=base_channel*2,
                out_channels=3,
                bias=False,
                kernel_size=4, stride=2, padding=1),

            nn.Tanh() # 32
        )

    def forward(self, x):
        return self.network(x.unsqueeze(-1).unsqueeze(-1))


class UpsampleGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        base_channel = 64
        self.network = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(
                in_channels=latent_dim,
                out_channels=base_channel*8,
                bias=False,
                kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*8),

            nn.ReLU(True), # 4

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=base_channel*8,
                out_channels=base_channel*4,
                bias=False,
                kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*4),
            nn.ReLU(True), # 8

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=base_channel*4,
                out_channels=base_channel*2,
                bias=False,
                kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=base_channel*2),
            nn.ReLU(True), # 16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=base_channel*2,
                out_channels=3,
                kernel_size=3, padding=1),
            nn.Tanh() # 32
        )

    def forward(self, x):
        return self.network(x.unsqueeze(-1).unsqueeze(-1))
