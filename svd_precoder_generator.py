import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from parameters import *
from utils import *


# run this file to generate svd precoder in each porint and samples using origin dataset

def main():
    f = scio.loadmat('./data/train_new.mat')
    data = f['train']  # f['test']
    channel_data = data[0, 0]['CSI']
    channel_data_train = np.zeros([N_point * N_train, N_rx, N_tx, N_f, 2])
    channel_data_test = np.zeros([N_point * N_test, N_rx, N_tx, N_f, 2])
    for index in range(N_point):
        channel_data_train[index * N_train:(index + 1) * N_train, ...] = channel_data[
                                                                         index * N_history:index * N_history + N_train,
                                                                         ...]
        channel_data_test[index * N_test:(index + 1) * N_test, ...] = channel_data[index * N_history + N_train:(
                                                                                                                       index + 1) * N_history,
                                                                      ...]

    PrecodingVector = DownPrecoding(channel_data_train)  # (N_dataset,Nf,NT,1)
    # PrecodingVector_real = np.real(PrecodingVector)
    # PrecodingVector_imag = np.imag(PrecodingVector)
    # PrecodingVector_torch = np.stack([PrecodingVector_real, PrecodingVector_imag], axis=-1)
    # PrecodingVector_save = PrecodingVector_torch.squeeze()
    # np.save("precoder_svd.npy", PrecodingVector_save)
    precoder_best_svd_eachpoint = []
    precoder_random_selected_eachpoint = []
    index_precoder_best_svdeachpoint = []
    channel_best_selected_svd = []
    data_rate_SVD_theorybest = []
    data_rate_SVD_eachpointbest = []
    f = open("./logs/generating_best_precoder4each_point_traindataset.txt", "w")

    for i in range(0, N_point):
        best_precoder_rate = 0
        best_precoder_index = 0
        random_select_precoder_index = np.random.randint(0, N_train)
        channel_org_point = channel_data_train[N_train * i: N_train * (i + 1), ...]
        precoder_point_all = PrecodingVector[N_train * i: N_train * (i + 1), ...]
        # find the best svd precoder of one point
        for index in range(len(precoder_point_all)):
            precoder_temp = precoder_point_all[index, ...].reshape(-1, N_f, N_tx, 1)
            precoder_temp_repeat = np.repeat(precoder_temp, N_train, axis=0)
            SubCH_gain_codeword_point = EqChannelGain(channel_org_point,
                                                      precoder_temp_repeat)  # (N_dataset,Nf,1),每个子载波上的gain
            data_rate_point = DataRate(SubCH_gain_codeword_point, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
            if data_rate_point > best_precoder_rate:
                best_precoder_rate = data_rate_point
                best_precoder_index = index
        precoder_best_svd = precoder_point_all[best_precoder_index].reshape(-1, N_f, N_tx, 1)
        precoder_random_selected = precoder_point_all[random_select_precoder_index].reshape(-1, N_f, N_tx, 1)
        channel_best_selected = channel_org_point[best_precoder_index]
        print(
            f'-------The best precoder of point:{i} is index {best_precoder_index} with score {best_precoder_rate} bps/Hz--------')
        precoder_best_svd_eachpoint.append(precoder_best_svd)
        precoder_random_selected_eachpoint.append(precoder_random_selected)
        data_rate_SVD_eachpointbest.append(best_precoder_rate)
        index_precoder_best_svdeachpoint.append(best_precoder_index)
        channel_best_selected_svd.append(channel_best_selected)

        # SVD best theory best
        SubCH_gain_codeword_best = EqChannelGain(channel_org_point, precoder_point_all)  # (N_dataset,Nf,1),每个子载波上的gain
        data_rate_best = DataRate(SubCH_gain_codeword_best, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
        print('-------The theory best score of SVD is %.8f bps/Hz--------\n' % data_rate_best)
        data_rate_SVD_theorybest.append(data_rate_best)
    # np.save('./data/index_precoder_svd_eachpoint.npy', np.array(index_precoder_best_svdeachpoint))
    np.save('./data/channel_selected_svd_eachpoint.npy', np.array(channel_best_selected_svd))

    plt.figure(1, dpi=300, figsize=(8, 6))
    plt.plot(data_rate_SVD_eachpointbest, label='best svd')
    plt.plot(data_rate_SVD_theorybest, label='iteration svd')
    plt.legend()
    plt.xlabel('number of points')
    plt.ylabel('data rates bps/Hz')
    plt.show()

    # precoder_random_selected_eachpoint = np.array(precoder_random_selected_eachpoint)
    # PrecodingVector_real = np.real(precoder_random_selected_eachpoint)
    # PrecodingVector_imag = np.imag(precoder_random_selected_eachpoint)
    # PrecodingVector_torch = np.stack([PrecodingVector_real, PrecodingVector_imag], axis=-1)
    # PrecodingVector_save = PrecodingVector_torch.squeeze()
    # np.save("./data/precoder_random_select_svd_eachpoint.npy", PrecodingVector_save)

    precoder_best_svd_eachpoint = np.array(precoder_best_svd_eachpoint)
    PrecodingVector_real = np.real(precoder_best_svd_eachpoint)
    PrecodingVector_imag = np.imag(precoder_best_svd_eachpoint)
    PrecodingVector_torch = np.stack([PrecodingVector_real, PrecodingVector_imag], axis=-1)
    PrecodingVector_save = PrecodingVector_torch.squeeze()
    np.save("./data/precoder_svd_eachpoint.npy", PrecodingVector_save)


def test():
    precoders_each_point = np.load('./data/precoder_svd_eachpoint.npy')
    precoders_each_point = precoders_each_point[..., 0] + 1j * precoders_each_point[..., 1]
    f = scio.loadmat('./data/train_new.mat')
    data = f['train']  # f['test']
    channel_data = data[0, 0]['CSI']
    PrecodingVector = DownPrecoding(channel_data)
    precoder_best_svd_eachpoint = []
    data_rate_SVD_theorybest = []
    data_rate_SVD_eachpointbest = []
    for i in range(0, N_point):
        channel_org_point = channel_data[N_history * i: N_history * (i + 1), ...]
        precoder_point_all = PrecodingVector[N_history * i: N_history * (i + 1), ...]
        # find the best svd precoder of one point
        precoder_temp = precoders_each_point[i, ...].reshape(-1, N_f, N_tx, 1)
        precoder_temp_repeat = np.repeat(precoder_temp, N_history, axis=0)
        SubCH_gain_codeword_point = EqChannelGain(channel_org_point,
                                                  precoder_temp_repeat)  # (N_dataset,Nf,1),每个子载波上的gain
        data_rate_point = DataRate(SubCH_gain_codeword_point, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
        data_rate_SVD_eachpointbest.append(data_rate_point)

        # SVD best theory best
        SubCH_gain_codeword_best = EqChannelGain(channel_org_point, precoder_point_all)  # (N_dataset,Nf,1),每个子载波上的gain
        data_rate_best = DataRate(SubCH_gain_codeword_best, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
        data_rate_SVD_theorybest.append(data_rate_best)
    plt.figure(1, dpi=300, figsize=(8, 6))
    plt.plot(data_rate_SVD_eachpointbest, label='best svd')
    plt.plot(data_rate_SVD_theorybest, label='iteration svd')
    plt.legend()
    plt.xlabel('number of points')
    plt.ylabel('data rates bps/Hz')
    plt.show()


def find_codebook_index_everysample():
    f = scio.loadmat('./data/train_new.mat')
    data = f['train']  # f['test']
    channel_data = data[0, 0]['CSI']
    channel_data_train = np.zeros([N_point * N_train, N_rx, N_tx, N_f, 2])
    channel_data_test = np.zeros([N_point * N_test, N_rx, N_tx, N_f, 2])
    for index in range(N_point):
        channel_data_train[index * N_train:(index + 1) * N_train, ...] = channel_data[
                                                                         index * N_history:index * N_history + N_train,
                                                                         ...]
        channel_data_test[index * N_test:(index + 1) * N_test, ...] = channel_data[index * N_history + N_train:(
                                                                                                                       index + 1) * N_history,
                                                                      ...]
    idx = np.arange(0, N_tx).reshape([16, 1])
    angle_set = np.arange(start=0, stop=180, step=0.1)
    angle_set = np.reshape(angle_set, [1, len(angle_set)]) / 180 * np.pi
    codebooks = np.exp(1j * np.pi * idx * np.cos(angle_set)).transpose()
    # np.save('DFTcodebook.npy', codebooks)
    precoder_index_codebook_eachsample = []
    # find the best codebook index precoder of one channel sample
    for i in range(len(channel_data_train)):
        channel_org_sample = channel_data_train[i, ...].reshape(-1, N_rx, N_tx, N_f, 2)
        channel_org_repeat = np.repeat(channel_org_sample, len(codebooks), axis=0)
        precoder_codebook = codebooks.reshape(-1, N_f, N_tx, 1)
        SubCH_gain_codeword_point = EqChannelGain(channel_org_repeat, precoder_codebook)
        SNR = SubCH_gain_codeword_point / sigma2_UE
        rate = np.mean(np.log2(1 + SNR), axis=-1)  ## rate
        best_precoder_index_codebook = np.argmax(rate)
        precoder_index_codebook_eachsample.append(best_precoder_index_codebook)
        print(f'the best precoder codebook index of channel{i} is {best_precoder_index_codebook}')
    np.save('./data/DFTcodebook_index_all_traindataset.npy', np.array(precoder_index_codebook_eachsample))


def find_codebook_index_everypoint():
    f = scio.loadmat('./data/train_new.mat')
    data = f['train']  # f['test']
    channel_data = data[0, 0]['CSI']
    channel_data_train = np.zeros([N_point * N_train, N_rx, N_tx, N_f, 2])
    channel_data_test = np.zeros([N_point * N_test, N_rx, N_tx, N_f, 2])
    for index in range(N_point):
        channel_data_train[index * N_train:(index + 1) * N_train, ...] = channel_data[index * N_history:index * N_history + N_train, ...]
        channel_data_test[index * N_test:(index + 1) * N_test, ...] = channel_data[index * N_history + N_train:(index + 1) * N_history, ...]
    codebook = np.load('./data/DFTcodebook.npy')  # [1800,16]
    beam_index = np.load('./data/DFTcodebook_index_all_traindataset.npy')  # [302400,]
    precoder_index_codebook_quzhongshu = []
    data_rate_codebook_theorybest = []
    data_rate_codebook_eachpointbest = []
    # PrecodingVector = codebook[beam_index]
    for i in range(0, N_point):
        best_precoder_rate = 0
        best_precoder_index = 0
        channel_org_point = channel_data_train[N_train * i: N_train * (i + 1), ...]
        # precoder_point_all = PrecodingVector[N_train * i: N_train * (i + 1), ...]
        beam_index_point_all = beam_index[N_train * i: N_train * (i + 1), ...]
        # find the best codebook index precoder of one point
        for index in beam_index_point_all:
            precoder_temp = codebook[index].reshape(-1, N_f, N_tx, 1)
            precoder_temp_repeat = np.repeat(precoder_temp, N_train, axis=0)
            SubCH_gain_codeword_point = EqChannelGain(channel_org_point,
                                                      precoder_temp_repeat)  # (N_dataset,Nf,1),每个子载波上的gain
            data_rate_point = DataRate(SubCH_gain_codeword_point, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
            if data_rate_point > best_precoder_rate:
                best_precoder_rate = data_rate_point
                best_precoder_index = index
        # precoder_best_svd = precoder_point_all[best_precoder_index].reshape(-1, N_f, N_tx, 1)
        print(
            f'-------The best precoder of point:{i} is index {best_precoder_index} with score {best_precoder_rate} bps/Hz--------')
        precoder_index_codebook_quzhongshu.append(best_precoder_index)
        data_rate_codebook_eachpointbest.append(best_precoder_rate)
        # SVD best theory best
        # SubCH_gain_codeword_best = EqChannelGain(channel_org_point, precoder_point_all.reshape(-1, N_f, N_tx,
        #                                                                                        1))  # (N_dataset,Nf,1),每个子载波上的gain
        # data_rate_best = DataRate(SubCH_gain_codeword_best, sigma2_UE)  # 对每个样本每个子载波做香农公式的平均
        # print('-------The theory best score of SVD is %.8f bps/Hz--------\n' % data_rate_best)
        # data_rate_codebook_theorybest.append(data_rate_best)
    np.save('./data/DFTcodebook_index_quzhongshu_traindataset.npy', np.array(precoder_index_codebook_quzhongshu))


if __name__ == "__main__":
    main()
    # test()
    # find_codebook_index_everysample()
    # find_codebook_index_everypoint()

    # test1 = np.load('./data/DFTcodebook_index_all.npy')
    # test2 = np.load('./data/DFTcodebook_index_quzhongshu.npy')
    # test3 = np.load('./data/DFTcodebook.npy')

