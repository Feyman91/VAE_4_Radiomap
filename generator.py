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

# Run this py file to generate precoders'/channels' data with trained vae model

filename = 'train'
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


def generate(num_epochs, generate_precoder):
    if generate_precoder:
        num_epoch = num_epochs[1]
        print('VAE generating precoder:')
        vae = VAE_precoder(input_dim_precoder, hidden_dim, latent_dim_precoder)
        precoder_svd = np.load('./data/precoder_svd.npy')
        dataset_loader = torch.utils.data.DataLoader(precoder_svd, batch_size=len(precoder_svd), shuffle=False)
        f = open(
            f"./logs/generating_vae_precoder_beta{beta}_epoch{num_epoch}_latent_dim{latent_dim_precoder}_lr{learning_rate}.txt",
            "w")
        vae.load_state_dict(
            torch.load('./networks/vae_precoder_beta{}_epoch{}_latent_dim{}_from_{}_py.pt'.format(beta, num_epoch,
                                                                                                  latent_dim_precoder,
                                                                                                  filename), map_location=torch.device(devices)))

        print('Load vae from vae_precoder_beta{}_epoch{}_latent_dim{}_lr{}_from_{}_py.pt'.format(beta, num_epoch,
                                                                                                 latent_dim_precoder,
                                                                                                 learning_rate,
                                                                                                 filename))
    else:
        num_epoch = num_epochs[0]
        print('VAE generating channel:')
        vae = VAE(input_dim, hidden_dim, latent_dim)
        dataset_loader = torch.utils.data.DataLoader(channel_data, batch_size=len(channel_data), shuffle=False)
        f = open(
            f"./logs/generating_vae_channel_beta{beta}_epoch{num_epoch}_latent_dim{latent_dim}_lr{learning_rate}.txt",
            "w")
        vae.load_state_dict(torch.load(
            './networks/vae_channel_beta{}_epoch{}_latent_dim{}_from_{}_py.pt'.format(beta, num_epoch, latent_dim,
                                                                                      filename), map_location=torch.device(devices)))

        print('Load vae from vae_channel_beta{}_epoch{}_latent_dim{}_from_{}_py.pt'.format(beta, num_epoch, latent_dim,
                                                                                           filename))

    # Evaluation
    vae.eval()
    x_generated_final = []
    x_generated_final_KL = []
    x_generated_using_miu_mean = []
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_loader):
            data = data.to(devices)
            for i in range(0, N_point):
                data_temp = data[N_history * i: N_history * (i + 1), :]
                x_reconstructed, mu, log_var = vae(data_temp)
                z_no_repara = mu
                z_mean = torch.mean(z_no_repara, dim=0)
                # generated_using_miu_mean = vae.decoder(z_mean)
                var = torch.mean(log_var, dim=0)
                x_generated = vae.decoder(z_no_repara)
                # 这里decoder为什么只输入了均值
                MSE_min = 111110
                min_index = 0
                for j in range(len(z_no_repara)):
                    MSE = nn.functional.mse_loss(z_mean, z_no_repara[j], reduction='sum')
                    MSE = MSE + nn.functional.mse_loss(var, log_var[j], reduction='sum')
                    # print(j, MSE)
                    if MSE < MSE_min:
                        MSE_min = MSE
                        min_index = j
                # print(min_index)

                mus = mu
                log_vars = log_var
                KL_min = 111110
                min_index_KL = 0
                for j in range(len(mus)):
                    KL = 0
                    for k in range(len(mus)):
                        if k < j:
                            KL = KL + KL4Norm(mus[k], log_vars[k], mus[j], log_vars[j])
                        elif k > j:
                            KL = KL + KL4Norm(mus[j], log_vars[j], mus[k], log_vars[k])
                    if KL < KL_min:
                        KL_min = KL
                        min_index_KL = j
                # print(min_index, min_index_KL)
                x_generated_final.append(x_generated[min_index].detach().numpy())
                x_generated_final_KL.append(x_generated[min_index_KL].detach().numpy())
                # x_generated_using_miu_mean.append(generated_using_miu_mean.detach().numpy())
                # if i % 10 == 0:
                print(f'generating point {i} with MSE min_index: {min_index}\tKL min_index_kl: {min_index_KL}', file=f)
                print(f'generating point {i} with MSE min_index: {min_index}\tKL min_index_kl: {min_index_KL}')
            if generate_precoder:
                np.save(f'./generate/precoder/Precoder_generated_final_{filename}.npy', x_generated_final)
                np.save(f'./generate/precoder/Precoder_generated_final_KL_{filename}.npy', x_generated_final_KL)
                # np.save(f'./generate/precoder/Precoder_generated_using_miu_mean_{filename}.npy', x_generated_using_miu_mean)
            else:
                np.save(f'./generate/channel/H_generated_final_{filename}.npy', x_generated_final)
                np.save(f'./generate/channel/H_generated_final_KL_{filename}.npy', x_generated_final_KL)
                # np.save(f'./generate/precoder/H_generated_using_miu_mean_{filename}.npy', x_generated_using_miu_mean)



if __name__ == "__main__":
    for_precoder = True
    if for_precoder:
        print('##############precoder VAE Net#############')
    else:
        print('##############CSI VAE Net#############')
    generate(num_epochs, for_precoder)

