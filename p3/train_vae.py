#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.optim import Adam
from vae_model import VAE
from loader import get_data_loader


def compute_loss(inputs, outputs, mu, logvar):
    reconstruction_loss = nn.MSELoss(inputs, outputs, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
    return kl_loss + reconstruction_loss

def train_vae():

    batch_size = 64
    epochs = 100
    latent_dimension = 100

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # load data
    train_loader, valid_loader, test_loader = get_data_loader('data', batch_size)

    model = VAE(latent_dimension).to(device)
    
    optim = Adam(model.parameters(), lr=1e-4)

    for e in range(epochs):
        running_loss = 0 
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            model.zero_grad()
            outputs, mu, logvar = model(images)
            loss = compute_loss(images, outputs, mu, logvar)
            running_loss += loss
            loss.backward()
            optim.step()

        torch.save({
            'epoch': e,
            'model': model.state_dict(),
            'running_loss': running_loss,
            'optim': optim.state_dict(),
        }, "vae_checkpoint_{}.pth".format(e))
        print("Epoch: {} Loss: {}".format(e+1, running_loss.item()/len(train_loader)))




