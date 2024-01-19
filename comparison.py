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

# run this file to compare result using new generated data or model, etc.

filename = 'train'
f = scio.loadmat('./data/train_new.mat')
data = f['train']
location_data = data[0, 0]['loc']
channel_data = data[0, 0]['CSI']
channel_data_train = np.zeros([N_point*N_train, N_rx, N_tx, N_f, 2])
channel_data_test = np.zeros([N_point*N_test, N_rx, N_tx, N_f, 2])
for index in range(N_point):
    channel_data_train[index*N_train:(index+1)*N_train, ...] = channel_data[index*N_history:index*N_history+N_train, ...]
    channel_data_test[index*N_test:(index+1)*N_test, ...] = channel_data[index*N_history+N_train:(index+1)*N_history, ...]

network_dir = 'networks'
if not os.path.exists(network_dir):
    os.makedirs(network_dir)

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def Comparison():
    data_rate_SVD_Exhaustive_search = []
    data_rate_best_svd_eachpoint = []
    data_rate_oversampled_codebook = []
    data_rate_oversampled_codebook_mode = []
    data_rate_channel_VAE = []
    data_rate_channel_VAE_KL = []
    data_rate_precoder_VAE = []
    data_rate_precoder_VAE_KL = []
    data_rate_precoder_VAE_average_mean = []
    data_rate_vaemse_sub_vaekl_precoder = []
    data_rate_vaeavemean_sub_vaemse_precoder = []
    data_rate_vaemse_sub_vaekl_channel = []
    data_rate_channel_randomselect = []

    channel_generated_VAE = np.load(f'./generate/channel/best_ver/H_generated_final_{filename}.npy')
    channel_generated_VAE_KL = np.load(f'./generate/channel/best_ver/H_generated_final_KL_{filename}.npy')

    precoder_generated_VAE = np.load(f'./generate/precoder/best_result_precoder_fordrawing/Precoder_generated_final_{filename}.npy')
    precoder_generated_VAE_KL = np.load(f'./generate/precoder/best_result_precoder_fordrawing/Precoder_generated_final_KL_{filename}.npy')
    # precoder_generated_VAE_average_mean = np.load(
    #     f'./generate/precoder/Precoder_generated_using_miu_mean_{filename}.npy')

    DFTcodebook = np.load('./data/DFTcodebook.npy')
    # beam_index_all = np.load('./data/DFTcodebook_index_all.npy')
    beam_index_quzhongshu = np.load('./data/DFTcodebook_index_quzhongshu_traindataset.npy')
    precoder_dftcodebook = DFTcodebook[beam_index_quzhongshu]

    precoder_best_each_point = np.load('./data/precoder_svd_eachpoint_traindataset.npy')
    precoder_best_each_point = precoder_best_each_point[..., 0] + 1j * precoder_best_each_point[..., 1]

    precoder_random_selected_each_point = np.load('./data/precoder_random_select_svd_eachpoint_traindataset.npy')
    precoder_random_selected_each_point = precoder_random_selected_each_point[..., 0] + 1j * precoder_random_selected_each_point[..., 1]

    for i in range(0, N_point):
        channel_org = channel_data_test[N_test * i: N_test * (i + 1), ...]
        # channel_vae_repeat = np.repeat(channel_vae.reshape(-1, N_rx, N_tx, N_f, 2), N_history, axis=0)       # 重复40次
        # print(channel_avg_position.shape)

        # SVD Exhaustive search best each point and each sample(N_history)
        # PrecodingVector1 = DownPrecoding(channel_org)  # (N_dataset,Nf,NT,1)
        # SubCH_gain_codeword_1 = EqChannelGain(channel_org, PrecodingVector1)  # (N_dataset,Nf,1),每个子载波上的gain
        # data_rate1 = DataRate(SubCH_gain_codeword_1, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
        # print('\n -------The score of SVD Exhaustive search is %.8f bps/Hz--------' % data_rate1)
        # data_rate_SVD_Exhaustive_search.append(data_rate1)

        # SVD best each point
        precoder_repeat = np.repeat(precoder_best_each_point[i, ...].reshape(-1, N_f, N_tx, 1), N_test, axis=0)
        SubCH_gain_codeword_point = EqChannelGain(channel_org, precoder_repeat)  # (N_dataset,Nf,1),每个子载波上的gain
        data_rate_point = DataRate(SubCH_gain_codeword_point, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
        print('The score of VAE best each point precoder is %.8f bps/Hz' % data_rate_point)
        data_rate_best_svd_eachpoint.append(data_rate_point)

        # random selected represent precoder each point
        precoder_repeat_random = np.repeat(precoder_random_selected_each_point[i, ...].reshape(-1, N_f, N_tx, 1), N_test, axis=0)
        SubCH_gain_codeword_point_random = EqChannelGain(channel_org, precoder_repeat_random)  # (N_dataset,Nf,1),每个子载波上的gain
        data_rate9 = DataRate(SubCH_gain_codeword_point_random, sigma2_UE)
        print('The score of random select channel is %.8f bps/Hz' % data_rate9)
        data_rate_channel_randomselect.append(data_rate9)

        # 码本
        # PrecodingVector2 = DFTcodebook[beam_index_all[N_history * i : N_history * (i+1)], :]
        # PrecodingVector2 = np.reshape(PrecodingVector2, [-1, N_f, N_tx, 1])
        # SubCH_gain_codeword_2 = EqChannelGain(channel_org, PrecodingVector2)
        # data_rate2 = DataRate(SubCH_gain_codeword_2, sigma2_UE)
        # print('The score of oversampled codebook is %.8f bps/Hz' % data_rate2)
        # data_rate_oversampled_codebook.append(data_rate2)

        # 码本取众数
        precoder_point_all = precoder_dftcodebook[i]
        PrecodingVector3 = np.repeat(precoder_point_all.reshape(-1, N_f, N_tx, 1), N_test, axis=0)
        # print(PrecodingVector3.shape)
        SubCH_gain_codeword_3 = EqChannelGain(channel_org, PrecodingVector3)
        data_rate3 = DataRate(SubCH_gain_codeword_3, sigma2_UE)
        print('The score of mode of oversampled codebook is %.8f bps/Hz' % data_rate3)
        data_rate_oversampled_codebook_mode.append(data_rate3)

        # VAE生成的代表性信道 mse mean
        channel_VAE = channel_generated_VAE[i]
        channel_VAE_repeat = np.repeat(channel_VAE.reshape(-1, N_rx, N_tx, N_f, 2), N_test, axis=0)
        PrecodingVector4 = DownPrecoding(channel_VAE_repeat)
        SubCH_gain_codeword_4 = EqChannelGain(channel_org, PrecodingVector4)
        data_rate4 = DataRate(SubCH_gain_codeword_4, sigma2_UE)
        print('The score of VAE channel is %.8f bps/Hz' % data_rate4)
        data_rate_channel_VAE.append(data_rate4)

        # VAE生成的代表性信道 KL
        channel_VAE_KL = channel_generated_VAE_KL[i]
        channel_VAE_repeat_KL = np.repeat(channel_VAE_KL.reshape(-1, N_rx, N_tx, N_f, 2), N_test, axis=0)
        PrecodingVector5 = DownPrecoding(channel_VAE_repeat_KL)
        SubCH_gain_codeword_5 = EqChannelGain(channel_org, PrecodingVector5)
        data_rate5 = DataRate(SubCH_gain_codeword_5, sigma2_UE)
        print('The score of VAE channel KL is %.8f bps/Hz' % data_rate5)
        data_rate_channel_VAE_KL.append(data_rate5)
        vaemse_sub_vaekl_channel = data_rate4 - data_rate5
        data_rate_vaemse_sub_vaekl_channel.append(vaemse_sub_vaekl_channel)
        print(f'the sub value between mse and kl channel result is {vaemse_sub_vaekl_channel}')

        # VAE生成的代表性预编码 mse mean
        precoder_VAE = precoder_generated_VAE[i]
        precoder_VAE_repeat = np.repeat(precoder_VAE.reshape(-1, N_f, N_tx, 2), N_test, axis=0)
        precoder_VAE_repeat_complex = precoder_VAE_repeat[:, :, :, 0] + 1j * precoder_VAE_repeat[:, :, :, 1]
        PrecodingVector6 = np.reshape(precoder_VAE_repeat_complex, [-1, N_f, N_tx, 1])
        SubCH_gain_codeword_6 = EqChannelGain(channel_org, PrecodingVector6)
        data_rate6 = DataRate(SubCH_gain_codeword_6, sigma2_UE)
        print('The score of VAE precoder is %.8f bps/Hz' % data_rate6)
        data_rate_precoder_VAE.append(data_rate6)

        # VAE生成的代表性预编码 KL
        precoder_VAE_KL = precoder_generated_VAE_KL[i]
        precoder_VAE_KL_repeat = np.repeat(precoder_VAE_KL.reshape(-1, N_f, N_tx, 2), N_test, axis=0)
        precoder_VAE_KL_repeat_complex = precoder_VAE_KL_repeat[:, :, :, 0] + 1j * precoder_VAE_KL_repeat[:, :, :, 1]
        PrecodingVector7 = np.reshape(precoder_VAE_KL_repeat_complex, [-1, N_f, N_tx, 1])
        SubCH_gain_codeword_7 = EqChannelGain(channel_org, PrecodingVector7)
        data_rate7 = DataRate(SubCH_gain_codeword_7, sigma2_UE)
        print('The score of VAE precoder KL is %.8f bps/Hz' % data_rate7)
        data_rate_precoder_VAE_KL.append(data_rate7)
        vaemse_sub_vaekl_precoder = data_rate6 - data_rate7
        data_rate_vaemse_sub_vaekl_precoder.append(vaemse_sub_vaekl_precoder)
        print(f'the sub value between mse and kl precoder result is {vaemse_sub_vaekl_precoder}')

        # VAE生成的代表性预编码使用miu mean
        # precoder_VAE_average_mean = precoder_generated_VAE_average_mean[i]
        # precoder_VAE_average_mean_repeat = np.repeat(precoder_VAE_average_mean.reshape(-1, N_f, N_tx, 2), N_history, axis=0)
        # precoder_VAE_average_mean_repeat_complex = precoder_VAE_average_mean_repeat[:, :, :, 0] + 1j * precoder_VAE_average_mean_repeat[:, :, :, 1]
        # PrecodingVector8 = np.reshape(precoder_VAE_average_mean_repeat_complex, [-1, N_f, N_tx, 1])
        # SubCH_gain_codeword_8 = EqChannelGain(channel_org, PrecodingVector8)
        # data_rate8 = DataRate(SubCH_gain_codeword_8, sigma2_UE)
        # print('The score of VAE precoder KL is %.8f bps/Hz' % data_rate8)
        # data_rate_precoder_VAE_average_mean.append(data_rate8)
        # vaeavemean_sub_vaemse_precoder = data_rate8 - data_rate6
        # data_rate_vaeavemean_sub_vaemse_precoder.append(vaeavemean_sub_vaemse_precoder)
        # print(f'the sub value between avemean and mse precoder result is {vaeavemean_sub_vaemse_precoder}')
    # np.save('data_rate_SVD.npy', data_rate_SVD)
    # np.save('data_rate_oversampled_codebook', data_rate_oversampled_codebook)
    # np.save('data_rate_oversampled_codebook_mode', data_rate_oversampled_codebook_mode)
    # np.save('data_rate_channel_VAE', data_rate_channel_VAE)
    # np.save('data_rate_channel_VAE_KL', data_rate_channel_VAE_KL)
    # np.save('data_rate_precoder_VAE', data_rate_precoder_VAE)
    # np.save('data_rate_precoder_VAE_KL', data_rate_precoder_VAE_KL)
    # np.save(f'./fig/data_rate_best_svd_eachpoint_beta{beta}_latent_dim{latent_dim_precoder}_lr{learning_rate}', np.array(data_rate_best_svd_eachpoint))
    # np.save(f'./fig/data_rate_precoder_VAE_eachpoint_beta{beta}_latent_dim{latent_dim_precoder}_lr{learning_rate}', np.array(data_rate_precoder_VAE))
    # np.save(f'./fig/data_rate_precoder_VAE_KL_eachpoint_beta{beta}_latent_dim{latent_dim_precoder}_lr{learning_rate}', np.array(data_rate_precoder_VAE_KL))

    print(
        f'the average data rate of origin precoder svd eachpoint is {np.mean(np.array(data_rate_best_svd_eachpoint))}')

    print(f'the average data rate of random selected represented channel is {np.mean(np.array(data_rate_channel_randomselect))}')
    print(f'the average data rate of generated channel with VAE MSE is {np.mean(np.array(data_rate_channel_VAE))}')
    print(f'the average data rate of generated channel with VAE KL is {np.mean(np.array(data_rate_channel_VAE_KL))}')
    print(f'the average data rate of generated precoder with quzhongshu oversampled codebook is {np.mean(np.array(data_rate_oversampled_codebook_mode))}')
    print(f'the average data rate of generated precoder with VAE MSE is {np.mean(np.array(data_rate_precoder_VAE))}')
    print(f'the average data rate of generated precoder with VAE KL is {np.mean(np.array(data_rate_precoder_VAE_KL))}')

    # print(f'the average data rate of generated precoder with VAE average mean is {np.mean(np.array(data_rate_precoder_VAE_average_mean))}')

    # 找出用 vae mse 选出生成的precoder比用vae kl选出生成的precoder更好的datarate的index
    index_vaemse_bigger_vaekl_precoder = np.where(np.array(data_rate_vaemse_sub_vaekl_precoder) > 0)[0]
    value_vaemse_bigger_vaekl_precoder = np.array(data_rate_vaemse_sub_vaekl_precoder)[
        index_vaemse_bigger_vaekl_precoder]

    # 找出用 vae mse 选出生成的channel比用vae kl选出生成的channel更好的datarate的index
    # index_vaemse_bigger_vaekl_channel = np.where(np.array(data_rate_vaemse_sub_vaekl_channel) > 0)[0]
    # value_vaemse_bigger_vaekl_channel = np.array(data_rate_vaemse_sub_vaekl_channel)[index_vaemse_bigger_vaekl_channel]

    # 计算用vae mse/kl生成的precoder/channel的datarate与svd遍历的datarate的差
    data_rate_vaemse_sub_svdeach_precoder = np.array(data_rate_precoder_VAE) - np.array(data_rate_best_svd_eachpoint)
    data_rate_vaekl_sub_svdeach_precoder = np.array(data_rate_precoder_VAE_KL) - np.array(data_rate_best_svd_eachpoint)
    # data_rate_vaemse_sub_svdeach_channel = np.array(data_rate_channel_VAE) - np.array(data_rate_best_svd_eachpoint)
    # data_rate_vaekl_sub_svdeach_channel = np.array(data_rate_channel_VAE_KL) - np.array(data_rate_best_svd_eachpoint)

    # 计算用码本取众数生成的precoder的datarate与svd遍历的datarate的差
    data_rate_codebook_sub_svdeach_precoder = np.array(data_rate_oversampled_codebook_mode) - np.array(data_rate_best_svd_eachpoint)

    # 找出用 vae mse 选出生成的precoder中优于使用svd找出的代表每个point最佳preocder的index
    index_vaemse_bigger_svdeach_precoder = np.where(data_rate_vaemse_sub_svdeach_precoder > 0)[0]
    value_vaemse_bigger_svdeach_precoder = np.array(data_rate_precoder_VAE)[index_vaemse_bigger_svdeach_precoder]

    # 找出用 vae kl选出生成的precoder中优于使用svd找出的代表每个point最佳preocder的index
    index_vaekl_bigger_svdeach_precoder = np.where(data_rate_vaekl_sub_svdeach_precoder > 0)[0]
    value_vaekl_bigger_svdeach_precoder = np.array(data_rate_precoder_VAE_KL)[index_vaekl_bigger_svdeach_precoder]

    # 找出用 vae mse 选出生成的channel产生precoder中优于使用svd找出的代表每个point最佳preocder的index
    # index_vaemse_bigger_svdeach_channel = np.where(data_rate_vaemse_sub_svdeach_channel > 0)[0]
    # value_vaemse_bigger_svdeach_channel = np.array(data_rate_channel_VAE)[index_vaemse_bigger_svdeach_channel]

    # 找出用 vae kl 选出生成的channel产生precoder中优于使用svd找出的代表每个point最佳preocder的index
    # index_vaekl_bigger_svdeach_channel = np.where(data_rate_vaekl_sub_svdeach_channel > 0)[0]
    # value_vaekl_bigger_svdeach_channel = np.array(data_rate_channel_VAE_KL)[index_vaekl_bigger_svdeach_channel]

    plt.figure(dpi=450, figsize=(12, 6))
    # plt.plot(data_rate_SVD_Exhaustive_search, label='SVD Exhaustive search', linewidth=0.5, alpha=0.7)
    plt.plot(data_rate_best_svd_eachpoint, label='SVD best each point', linewidth=0.7, alpha=0.7)
    # plt.plot(data_rate_oversampled_codebook, label = 'Oversampled codebook')
    # plt.plot(data_rate_oversampled_codebook_mode, label='Mode of oversampled codebook', linewidth=0.7, alpha=0.7)
    plt.plot(data_rate_channel_VAE, label='VAE channel', linewidth=0.7, alpha=0.7)
    # plt.plot(data_rate_channel_VAE_KL, label='VAE channel KL', linewidth=0.7, alpha=0.7)
    # plt.plot(data_rate_precoder_VAE, label='VAE precoder', linewidth=0.7, alpha=0.7)
    # plt.plot(data_rate_precoder_VAE_KL, label='VAE precoder KL', linewidth=0.7, alpha=0.7)
    # plt.plot(data_rate_precoder_VAE_average_mean, label='VAE precoder KL', linewidth=0.7, alpha=0.7)
    # plt.plot(np.abs(data_rate_vaemse_sub_svdeach_precoder), label='data rate vaemse sub svdeach precoder', linewidth=0.7, alpha=0.7)
    # plt.plot(np.abs(data_rate_codebook_sub_svdeach_precoder), label='data rate codebook sub svdeach precoder', linewidth=0.7, alpha=0.7)
    # plt.plot(np.arange(len(data_rate_vaemse_sub_svdeach_precoder)), np.repeat(np.mean(np.abs(data_rate_vaemse_sub_svdeach_precoder)), 3780), label='ave data rate vaemse sub svdeach precoder')
    # plt.plot(np.arange(len(data_rate_codebook_sub_svdeach_precoder)), np.repeat(np.mean(np.abs(data_rate_codebook_sub_svdeach_precoder)), 3780), label='ave data rate codebook sub svdeach precoder')


    # plt.plot(data_rate_channel0, label='channel 0')

    plt.scatter(index_vaemse_bigger_svdeach_precoder, value_vaemse_bigger_svdeach_precoder, label='Precoder VAE MSE better than SVD', marker='^',
                alpha=0.7, s=35)
    # plt.scatter(index_vaemse_bigger_svdeach_channel, value_vaemse_bigger_svdeach_channel,
    #             label='channel VAE MSE better than SVD', marker='*',
    #             alpha=0.7, s=35)
    # plt.scatter(index_vaekl_bigger_svdeach, value_vaekl_bigger_svdeach, label='VAE KL better than SVD', marker='x', alpha=0.7, s=35)

    plt.legend(loc=1)
    plt.xlabel('number of points')
    plt.ylabel('data rates bps/Hz')
    plt.grid(True)

    # plt.twinx()
    # plt.bar(index_vaemse_bigger_vaekl, value_vaemse_bigger_vaekl, label='VAE MSE better than VAE KL', alpha=0.7, color='violet')
    # plt.ylabel('data rates VAE MSE subs VAE KL bps/Hz')
    # plt.legend(loc=2)

    plt.savefig(f'./fig/vae_precoder_avemean_beta{beta}_latent_dim{latent_dim_precoder}_lr{learning_rate}_result.png')
    plt.show()
    print('compute over')


if __name__ == "__main__":
    print('##############result analysis#############')
    Comparison()
