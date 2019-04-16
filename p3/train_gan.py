#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.optim import Adam
from gan_model import Generator, Discriminator
from loader import get_data_loader


def gradient_penalty(disc_model, real_images, fake_images):
    epsilon = real_images.new_empty(real_images.size())
    epsilon.uniform_()
    intermediate = epsilon * real_images  + (1 - epsilon) * fake_images
    intermediate.requires_grad = True
    outputs = disc_model(intermediate)
    grads = torch.autograd.grad(
        outputs,
        intermediate,
        outputs.new_ones(outputs.size()),
        create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1)**2).mean()


def train_gan():

    batch_size = 64
    epochs = 100
    disc_update = 1
    gen_update = 5
    latent_dimension = 100
    lambduh = 10

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # load data
    train_loader, valid_loader, test_loader = get_data_loader('data', batch_size)

    disc_model = Discriminator().to(device)
    gen_model = Generator(latent_dimension).to(device)
    
    disc_optim = Adam(disc_model.parameters(), lr=1e-4)
    gen_optim = Adam(gen_model.parameters(), lr=1e-4)

    for e in range(epochs):
        disc_loss = 0 
        gen_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            step = i+1
            if step % disc_update == 0:
                disc_model.zero_grad()
                # sample noise
                noise = torch.randn((batch_size, latent_dimension)).to(device)

                # loss on fake
                inputs = gen_model(noise).detach()
                outputs = disc_model(inputs)
                loss = outputs.mean()

                # loss on real
                outputs = disc_model(images)
                loss -= outputs.mean()

                # add gradient penalty
                loss += lambduh * gradient_penalty(disc_model, images, inputs)

                disc_loss += loss / batch_size
                loss.backward()
                disc_optim.step()

            if step % gen_update == 0:
                gen_model.zero_grad()

                noise = torch.randn((batch_size, latent_dimension)).to(device)
                inputs = gen_model(noise)
                outputs = disc_model(inputs)
                loss = -outputs.mean()  
                
                gen_loss += loss / batch_size
                disc_model.zero_grad()
                loss.backward()
                disc_model.zero_grad()
                gen_optim.step()
        print("Epoch: {} Disc loss: {}".format(e+1, disc_loss.item()))
        print("Epoch: {} Gen loss: {}".format(e+1, gen_loss.item()))




