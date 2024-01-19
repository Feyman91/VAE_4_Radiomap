import numpy as np
import scipy.io as scio
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from utils import *
from model import *
from parameters import *
from generator import generate

# Run this py file to training precoders'/channels' vae model

filename = 'train3'
f = scio.loadmat('./data/train_new.mat')
data = f['train']
location_data = data[0, 0]['loc']
channel_data = data[0, 0]['CSI']

network_dir = 'networks'
if not os.path.exists(network_dir):
    os.makedirs(network_dir)

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def train(num_epochs, train_precoder):
    if train_precoder:
        num_epoch = num_epochs[1]
        vae = VAE_precoder(input_dim_precoder, hidden_dim, latent_dim_precoder)
        print('VAE training for precoder, beta = {}, latent_dim = {}, epoch = {}'.format(beta, latent_dim_precoder,
                                                                                         num_epoch))

        precoder_svd = np.load('./data/precoder_svd.npy')
        train_loader = torch.utils.data.DataLoader(precoder_svd, batch_size=batch_size, shuffle=True)
        f = open("./logs/training_vae_precoder_beta{}_epoch{}_latent_dim{}_lr{}_{}.txt".format(beta, num_epoch,
                                                                                            latent_dim_precoder,
                                                                                            learning_rate, filename), "w")
        input_d = input_dim_precoder

    else:
        num_epoch = num_epochs[0]
        vae = VAE(input_dim, hidden_dim, latent_dim)
        print('VAE training for channel, beta = {}, latent_dim = {}, epoch = {}'.format(beta, latent_dim, num_epoch))

        train_loader = torch.utils.data.DataLoader(channel_data, batch_size=batch_size, shuffle=True)
        f = open("./logs/training_vae_channel_beta{}_epoch{}_latent_dim{}_lr{}_{}.txt".format(beta, num_epoch, latent_dim,
                                                                                           learning_rate, filename), "w")
        input_d = input_dim

    vae.to(devices)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        vae.train()
        train_loss = 0
        train_loss_recons = 0
        train_loss_KL = 0
        for batch_idx, data in enumerate(train_loader):
            # print(batch_idx)
            # data = data.view(-1, input_dim).to(device)
            data = data.to(devices)
            optimizer.zero_grad()

            x_reconstructed, mu, log_var = vae(data)
            loss, loss_recons, loss_KL = vae_loss(data.view(-1, input_d), x_reconstructed, mu, log_var, beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_recons += loss_recons.item()
            train_loss_KL += loss_KL.item()

        print(
            f"Epoch [{epoch + 1}/{num_epoch}], Total Loss: {train_loss / len(train_loader.dataset):.4f}, Reconstruction Loss: {train_loss_recons / len(train_loader.dataset):.4f}"
            f", KL Loss: {train_loss_KL / len(train_loader.dataset):.4f}", file=f)
        print(
            f"Epoch [{epoch + 1}/{num_epoch}], Total Loss: {train_loss / len(train_loader.dataset):.4f}, Reconstruction Loss: {train_loss_recons / len(train_loader.dataset):.4f}"
            f", KL Loss: {train_loss_KL / len(train_loader.dataset):.4f}")

    if train_precoder:
        torch.save(vae.state_dict(),
                   './networks/vae_precoder_beta{}_epoch{}_latent_dim{}_from_{}_py.pt'.format(beta, num_epoch,
                                                                                              latent_dim_precoder,
                                                                                              filename))
    else:
        torch.save(vae.state_dict(),
                   './networks/vae_channel_beta{}_epoch{}_latent_dim{}_from_{}_py.pt'.format(beta, num_epoch,
                                                                                             latent_dim, filename))



if __name__ == "__main__":
    for_precoder = False
    if for_precoder:
        print('##############precoder VAE Net#############')
    else:
        print('##############CSI VAE Net#############')
    train(num_epochs, for_precoder)
    generate(num_epochs, for_precoder)


