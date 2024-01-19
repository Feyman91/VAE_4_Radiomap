from parameters import *


def DownPrecoding(channel_est):  ### Optional
    ### estimated channel
    HH_est = np.reshape(channel_est, (-1, N_rx, N_tx, N_f, 2))  ## Rx, Tx, Subcarrier, RealImag
    HH_complex_est = HH_est[:, :, :, :, 0] + 1j * HH_est[:, :, :, :, 1]  ## Rx, Tx, Subcarrier
    HH_complex_est = np.transpose(HH_complex_est, [0, 3, 1, 2])  # N_dataset, Nf, Nr,NT

    ### precoding based on the estimated channel
    MatRx, MatDiag, MatTx = np.linalg.svd(HH_complex_est, full_matrices=True)  ## SVD,np返回的是V的共轭转置
    # print(MatTx.shape)
    PrecodingVector = np.conj(MatTx[:, :, 0, :])  ## The best eigenvector (MRT transmission)   # 最大的一列的SVD的V，取了共轭，得到真正的V
    PrecodingVector = np.reshape(PrecodingVector, (-1, N_f, N_tx, 1))  # N_dataset,Nf,NT,1，转置
    return PrecodingVector


def DataRate(h_sub_gain, sigma2_UE):  ### Score
    SNR = h_sub_gain / sigma2_UE
    Rate = np.log2(1 + SNR)  ## rate
    Rate_OFDM = np.mean(Rate, axis=-1)  ###  averaging over subcarriers
    Rate_OFDM_mean = np.mean(Rate_OFDM)  ### averaging over CSI samples
    return Rate_OFDM_mean


def EqChannelGain(channel, PrecodingVector):
    ### The authentic CSI
    HH = np.reshape(channel, (-1, N_rx, N_tx, N_f, 2))  ## Rx, Tx, Subcarrier, RealImag
    HH_complex = HH[:, :, :, :, 0] + 1j * HH[:, :, :, :, 1]  ## Rx, Tx, Subcarrier
    HH_complex = np.transpose(HH_complex, [0, 3, 1, 2])

    ### Power Normalization of the precoding vector
    Power = np.matmul(np.transpose(np.conj(PrecodingVector), (0, 1, 3, 2)), PrecodingVector)  # 矩阵乘法，W'*W (1,1)
    PrecodingVector = PrecodingVector / np.sqrt(Power)  # norm

    ### Effective channel gain
    R = np.matmul(HH_complex, PrecodingVector)  # H*W, (N_dataset, Nf, Nr,1)
    R_conj = np.transpose(np.conj(R), (0, 1, 3, 2))  # 共轭转置(N_dataset, Nf, 1,Nr)
    h_sub_gain = np.matmul(R_conj, R)  # R'R
    h_sub_gain = np.reshape(np.absolute(h_sub_gain), (-1, N_f))  ### channel gain of SC_num subcarriers，每个子载波上的大小
    return h_sub_gain

