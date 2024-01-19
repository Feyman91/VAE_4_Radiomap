from parameters import *
from utils import *
import scipy.io as scio


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
        channel_data_test[index * N_test:(index + 1) * N_test, ...] = channel_data[
                                                                      index * N_history + N_train:(
                                                                                                              index + 1) * N_history,
                                                                      ...]

    ### estimated channel
    HH_est = np.reshape(channel_data_train, (-1, N_rx, N_tx, N_f, 2))  ## Rx, Tx, Subcarrier, RealImag
    HH_complex_est = HH_est[:, :, :, :, 0] + 1j * HH_est[:, :, :, :, 1]  ## Rx, Tx, Subcarrier
    HH_complex_est = np.transpose(HH_complex_est, [0, 3, 1, 2])  # N_dataset, Nf, Nr,NT

    ### precoding based on the estimated channel
    MatRx, MatDiag, MatTx = np.linalg.svd(HH_complex_est, full_matrices=True)  ## SVD,np返回的是V的共轭转置
    PrecodingVector = np.conj(MatTx[:, :, 0, :])  ## The best eigenvector (MRT transmission)   # 最大的一列的SVD的V，取了共轭，得到真正的V
    PrecodingVector = np.reshape(PrecodingVector, (-1, N_f, N_tx, 1))  # N_dataset,Nf,NT,1，转置

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
