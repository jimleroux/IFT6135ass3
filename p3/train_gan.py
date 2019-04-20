#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.optim import Adam
from gan_model import Generator, Discriminator, weights_init
from loader import get_data_loader
from vae_model import VAE


def gradient_penalty(disc_model, real_images, fake_images, device):
    epsilon = torch.rand(real_images.shape[0], 1, device=device)
    epsilon = epsilon.expand(real_images.shape[0], 3*32*32).view(real_images.shape[0], 3, 32, 32)
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
    disc_model.apply(weights_init)
    gen_model.apply(weights_init)

    # load vae weights and fine tune on them
    vae_model = '../vae/checkpoint_99.pth'
    vae_checkpoint = torch.load(vae_model)
    vae_model = VAE(latent_dimension)
    vae_model.load_state_dict(vae_checkpoint['model'])
    disc_model.conv.load_state_dict(vae_model.encoder.state_dict())
    gen_model.load_state_dict(vae_model.decoder.state_dict())
    del vae_model
    
    disc_optim = Adam(disc_model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    gen_optim = Adam(gen_model.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for e in range(epochs):
        disc_loss = 0 
        gen_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            b_size = images.shape[0]
            step = i+1
            if step % disc_update == 0:
                disc_model.zero_grad()
                # sample noise
                noise = torch.randn((b_size, latent_dimension), device=device)

                # loss on fake
                inputs = gen_model(noise).detach()
                f_outputs = disc_model(inputs)
                loss = f_outputs.mean()

                # loss on real
                r_outputs = disc_model(images)
                loss -= r_outputs.mean()

                # add gradient penalty
                loss += lambduh * gradient_penalty(disc_model, images, inputs, device)

                disc_loss += loss
                loss.backward()
                disc_optim.step()

            if step % gen_update == 0:
                gen_model.zero_grad()

                noise = torch.randn((b_size, latent_dimension)).to(device)
                inputs = gen_model(noise)
                outputs = disc_model(inputs)
                loss = -outputs.mean()  
                
                gen_loss += loss
                loss.backward()
                gen_optim.step()

        torch.save({
            'epoch': e,
            'disc_model': disc_model.state_dict(),
            'gen_model': gen_model.state_dict(),
            'disc_loss': disc_loss,
            'gen_loss': gen_loss,
            'disc_optim': disc_optim.state_dict(),
            'gen_optim': gen_optim.state_dict()
        }, "upsample/checkpoint_{}.pth".format(e))
        print("Epoch: {} Disc loss: {}".format(e+1, disc_loss.item()/len(train_loader)))
        print("Epoch: {} Gen loss: {}".format(e+1, gen_loss.item()/len(train_loader)))




