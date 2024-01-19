import numpy as np
import scipy.io as scio
from parameters import *


precoder_generated_VAE = np.load('./generate/precoder/best_result_precoder_fordrawing/Precoder_generated_final_train.npy')
precoder_svd = np.load('./data/precoder_svd_eachpoint.npy')
DFTcodebook = np.load('./data/DFTcodebook.npy')
beam_index_quzhongshu = np.load('./data/DFTcodebook_index_quzhongshu_traindataset.npy')
precoder_dftcodebook = DFTcodebook[beam_index_quzhongshu]

precoder_generated_VAE = precoder_generated_VAE.reshape(-1, 16, 2)
precoder_generated_VAE = precoder_generated_VAE[:, :, 0] + 1j * precoder_generated_VAE[:, :, 1]
precoder_svd = precoder_svd[:, :, 0] + 1j * precoder_svd[:, :, 1]

precoder_saved_svd = precoder_svd[:, :]
precoder_saved_vae = precoder_generated_VAE[:, :]

mdic = {"precoder_svd": precoder_saved_svd, "precoder_vae": precoder_saved_vae, "precoder_codebook": precoder_dftcodebook} # 创建一个字典
scio.savemat("F:/ph.d/Magazine/vae/matlab/precoder_test.mat", mdic) # 保存字典到文件中

#
# def get_channel_singular_value(channel_est):
#     ## Rx, Tx, Subcarrier
#     HH_complex_est = np.transpose(channel_est, [0, 3, 1, 2])  # N_dataset, Nf, Nr,NT
#
#     ### precoding based on the estimated channel
#     MatRx, MatDiag, MatTx = np.linalg.svd(HH_complex_est, full_matrices=True)  ## SVD,np返回的是V的共轭转置
#     return MatDiag
#
#
# f = scio.loadmat('./data/train_new.mat')
# data = f['train']  # f['test']
# channel_data = data[0, 0]['CSI']
# channel_svd_selected = np.load('./data/channel_selected_svd_eachpoint_traindataset.npy').reshape(
#     [-1, N_rx, N_tx, N_f, 2]).squeeze()
# channel_vae_mse = np.load('./generate/channel/best_ver/H_generated_final_train.npy').reshape([-1, N_rx, N_tx, N_f, 2]).squeeze()
# channel_vae_mse_kl = np.load('./generate/channel/best_ver/H_generated_final_KL_train.npy').reshape(
#     [-1, N_rx, N_tx, N_f, 2]).squeeze()
#
# channel_svd_selected = channel_svd_selected[..., 0] + 1j * channel_svd_selected[..., 1]
# channel_vae_mse = channel_vae_mse[..., 0] + 1j * channel_vae_mse[..., 1]
# channel_vae_mse_kl = channel_vae_mse_kl[..., 0] + 1j * channel_vae_mse_kl[..., 1]
#
# codebooks = np.load('./data/DFTcodebook.npy').transpose()
#
# PAS_svd_selected = np.matmul(channel_svd_selected, codebooks)
# PAS_vae_mse = np.matmul(channel_vae_mse, codebooks)
# PAS_vae_mse_kl = np.matmul(channel_vae_mse_kl, codebooks)
#
# PAS_svd_selected_norm = np.linalg.norm(PAS_svd_selected, axis=1)
# PAS_vae_mse_norm = np.linalg.norm(PAS_vae_mse, axis=1)
# PAS_vae_mse_kl_norm = np.linalg.norm(PAS_vae_mse_kl, axis=1)
#
# mdic = {"PAS_svd_selected": PAS_svd_selected, "PAS_vae_mse": PAS_vae_mse, "PAS_vae_mse_kl": PAS_vae_mse_kl}  # 创建一个字典
# mdic_norm = {"PAS_svd_selected_norm": PAS_svd_selected_norm, "PAS_vae_mse_norm": PAS_vae_mse_norm, "PAS_vae_mse_kl_norm": PAS_vae_mse_kl_norm}  # 创建一个字典
# scio.savemat("F:/ph.d/Magazine/vae/matlab/channel_pas.mat", mdic)  # 保存字典到文件中
# scio.savemat("F:/ph.d/Magazine/vae/matlab/channel_pas_norm.mat", mdic_norm)  # 保存字典到文件中

# diag_svd_channel = get_channel_singular_value(channel_svd_selected)
# diag_vae_mse_channel = get_channel_singular_value(channel_vae_mse)
# diag_vae_kl_channel = get_channel_singular_value(channel_vae_mse_kl)


