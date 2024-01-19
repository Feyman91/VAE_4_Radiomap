import os
import torch
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
devices = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

np.random.seed(2023)
torch.manual_seed(2023)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2023)

N_f = 1                     ## subcarrier number
N_tx = 16                   ## Tx antenna number
N_rx = 4                    ## Rx antenna number
N_point = 3780              # number of user point
N_history = 100             # number of slot OFDM samples
N_train = 80                # number of training samples per user point
N_test = 20                 # number of testing samples per user point
learning_rate = 5e-4
batch_size = 128
sigma2_UE = 0.1             # noise var
num_epochs = [110, 70]      # num_epochs[0] represents channel vae training epochs, num_epoch[1] represents precoder vae training epochs

input_dim_channel = N_tx * N_rx * 2     # input_dim_channel used in channel vae model input
input_dim = input_dim_channel           # input_dim used in channel vae model input, same as input_dim_channel
input_dim_precoder = N_tx * 2           # input_dim_precoder used in precoder vae model input

hidden_dim = [400, 128]     # input_dim => hidden_dim[0] => hidden_dim[1] => latend => hidden_dim[1] => hidden_dim[0] => input_dim

latent_dim = 60             # latent size used in channel vae model
latent_dim_precoder = 20    # latent size used in precoder vae model

beta = 0.01  # used for kl loss and reconstruct loss 's balance, the bigger it is, the consideration of kl loss weights higher
