import torch

import numpy as np
import scipy.io as scio
import datetime
from model import ANN_TypeI
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from utils import DataRateLoss, train_step, verify_step_time_eval, verify_step_space_eval, check_directory
import torch.utils.data as Data

N_aug = 100

f = scio.loadmat('./data/train_new.mat')
data = f['train']
location_data = data[0, 0]['loc']
channel_data = data[0, 0]['CSI']
location_train = location_data[::100, ...]
location_train[:, 2] = np.array(list(range(0, 3780)))
channel_aug = np.load(f"./data/generate_iter_{int(N_aug/100)}_equal_{N_aug}/channel_generated_reconstructed_generated30.npy")
print(channel_aug.shape)

# first part moving horizontal, Y change, X remain constant from [242.423004，441.170990], to [242.423004，483.170990]
index_start_point_stage1 = np.argwhere(np.all(location_train[:, 0:2] == np.array([242.423004,441.170990], dtype='float32'), axis=1)).squeeze()
print(f'Stage 1:  start location (X: {location_train[index_start_point_stage1,:].squeeze()[0]}  '
      f'Y: {location_train[index_start_point_stage1,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_start_point_stage1,:].squeeze()[2]})')

index_end_point_stage1 = np.argwhere(np.all(location_train[:, 0:2] == np.array([242.423004, 483.170990], dtype='float32'), axis=1)).squeeze()
print(f'Stage 1:  end location (X: {location_train[index_end_point_stage1,:].squeeze()[0]}  '
      f'Y: {location_train[index_end_point_stage1,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_end_point_stage1,:].squeeze()[2]})')

index_totalPoint_select_stage_1 = np.arange(index_start_point_stage1, index_end_point_stage1+1, 30)
print(f'Stage 1:  all point index ( X stable Y change): {index_totalPoint_select_stage_1}, length {np.size(index_totalPoint_select_stage_1)}')
index_seen_point_stage_1 = index_totalPoint_select_stage_1[::5]
index_unseen_point_stage_1 = np.setdiff1d(index_totalPoint_select_stage_1, index_seen_point_stage_1)
print(f'Stage 1:  Seen point {index_seen_point_stage_1}, length {np.size(index_seen_point_stage_1)}')
print(f'Stage 1:  Unseen point {index_unseen_point_stage_1}, length {np.size(index_unseen_point_stage_1)}\n\n')


# second part moving vertical, X change, Y remain constant from [242.423004，483.170990], to [245.423004，483.170990]
index_start_point_stage2 = np.argwhere(np.all(location_train[:, 0:2] == np.array([242.423004, 483.170990], dtype='float32'), axis=1)).squeeze()
print(f'Stage 2:  start location (X: {location_train[index_start_point_stage2,:].squeeze()[0]}  '
      f'Y: {location_train[index_start_point_stage2,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_start_point_stage2,:].squeeze()[2]})')

index_end_point_stage2 = np.argwhere(np.all(location_train[:, 0:2] == np.array([245.423004, 483.170990], dtype='float32'), axis=1)).squeeze()
print(f'Stage 2:  end location (X: {location_train[index_end_point_stage2,:].squeeze()[0]}  '
      f'Y: {location_train[index_end_point_stage2,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_end_point_stage2,:].squeeze()[2]})')

index_totalPoint_select_stage_2 = np.arange(index_start_point_stage2, index_end_point_stage2+1)
print(f'Stage 2:  all point index ( X stable Y change): {index_totalPoint_select_stage_2}, length {np.size(index_totalPoint_select_stage_2)}')
index_seen_point_stage_2 = index_totalPoint_select_stage_2[::5]
index_unseen_point_stage_2 = np.setdiff1d(index_totalPoint_select_stage_2, index_seen_point_stage_2)
print(f'Stage 2:  Seen point {index_seen_point_stage_2}, length {np.size(index_seen_point_stage_2)}')
print(f'Stage 2:  Unseen point {index_unseen_point_stage_2}, length {np.size(index_unseen_point_stage_2)}\n\n')


# Third part moving horizontal, Y change, X remain constant from [245.423004, 483.170990], to [245.423004, 525.171021]
index_start_point_stage3 = np.argwhere(np.all(location_train[:, 0:2] == np.array([245.423004, 483.170990], dtype='float32'), axis=1)).squeeze()
print(f'Stage 3:  start location (X: {location_train[index_start_point_stage3,:].squeeze()[0]}  '
      f'Y: {location_train[index_start_point_stage3,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_start_point_stage3,:].squeeze()[2]})')

index_end_point_stage3 = np.argwhere(np.all(location_train[:, 0:2] == np.array([245.423004, 525.171021], dtype='float32'), axis=1)).squeeze()
print(f'Stage 3:  end location (X: {location_train[index_end_point_stage3,:].squeeze()[0]}  '
      f'Y: {location_train[index_end_point_stage3,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_end_point_stage3,:].squeeze()[2]})')

index_totalPoint_select_stage_3 = np.arange(index_start_point_stage3, index_end_point_stage3+1, 30)
print(f'Stage 3:  all point index ( X stable Y change): {index_totalPoint_select_stage_3}, length {np.size(index_totalPoint_select_stage_3)}')
index_seen_point_stage_3 = index_totalPoint_select_stage_3[::5]
index_unseen_point_stage_3 = np.setdiff1d(index_totalPoint_select_stage_3, index_seen_point_stage_3)
print(f'Stage 3:  Seen point {index_seen_point_stage_3}, length {np.size(index_seen_point_stage_3)}')
print(f'Stage 3:  Unseen point {index_unseen_point_stage_3}, length {np.size(index_unseen_point_stage_3)}\n\n')


# Fourth part moving vertical, X change, Y remain constant from [245.423004, 525.171021], to [248.223007, 525.171021]
index_start_point_stage4 = np.argwhere(np.all(location_train[:, 0:2] == np.array([245.423004, 525.171021], dtype='float32'), axis=1)).squeeze()
print(f'Stage 4:  start location (X: {location_train[index_start_point_stage4,:].squeeze()[0]}  '
      f'Y: {location_train[index_start_point_stage4,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_start_point_stage4,:].squeeze()[2]})')

index_end_point_stage4 = np.argwhere(np.all(location_train[:, 0:2] == np.array([248.223007, 525.171021], dtype='float32'), axis=1)).squeeze()
print(f'Stage 4:  end location (X: {location_train[index_end_point_stage4,:].squeeze()[0]}  '
      f'Y: {location_train[index_end_point_stage4,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_end_point_stage4,:].squeeze()[2]})')

index_totalPoint_select_stage_4 = np.arange(index_start_point_stage4, index_end_point_stage4+1)
print(f'Stage 4:  all point index ( X stable Y change): {index_totalPoint_select_stage_4}, length {np.size(index_totalPoint_select_stage_4)}')
indices = [i for i in range(0, len(index_totalPoint_select_stage_4), 5)]
indices.append(len(index_totalPoint_select_stage_4)-1)
index_seen_point_stage_4 = index_totalPoint_select_stage_4[indices]
index_unseen_point_stage_4 = np.setdiff1d(index_totalPoint_select_stage_4, index_seen_point_stage_4)
print(f'Stage 4:  Seen point {index_seen_point_stage_4}, length {np.size(index_seen_point_stage_4)}')
print(f'Stage 4:  Unseen point {index_unseen_point_stage_4}, length {np.size(index_unseen_point_stage_4)}\n\n')


# Fifth part moving Right deviation, X change, Y change from [248.223007, 525.171021], to [244.223007, 549.171021]
index_start_point_stage5 = np.argwhere(np.all(location_train[:, 0:2] == np.array([248.223007, 525.171021], dtype='float32'), axis=1)).squeeze()
print(f'Stage 5:  start location (X: {location_train[index_start_point_stage5,:].squeeze()[0]}  '
      f'Y: {location_train[index_start_point_stage5,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_start_point_stage5,:].squeeze()[2]})')

index_end_point_stage5 = np.argwhere(np.all(location_train[:, 0:2] == np.array([244.223007, 549.171021], dtype='float32'), axis=1)).squeeze()
print(f'Stage 5:  end location (X: {location_train[index_end_point_stage5,:].squeeze()[0]}  '
      f'Y: {location_train[index_end_point_stage5,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_end_point_stage5,:].squeeze()[2]})')

location_x_stage_5 = location_train[index_end_point_stage5:index_end_point_stage5+21, :][:, 0][::-1]
location_y_stage_5 = location_train[index_start_point_stage5:index_end_point_stage5+30:30, :][:, 1]
location_xy_stage_5 = np.stack((location_x_stage_5, location_y_stage_5), axis=1)
indices = []

for element in location_xy_stage_5:
    matching_indices = np.where((location_train[:, 0:2] == element).all(axis=1))[0]
    indices.append(matching_indices)
# 将结果转化为numpy数组
index_totalPoint_select_stage_5 = np.array(indices).squeeze()
print(f'Stage 5:  all point index ( X change Y change): {index_totalPoint_select_stage_5}, length {np.size(index_totalPoint_select_stage_5)}')
index_seen_point_stage_5 = index_totalPoint_select_stage_5[::5]
index_unseen_point_stage_5 = np.setdiff1d(index_totalPoint_select_stage_5, index_seen_point_stage_5)
print(f'Stage 5:  Seen point {index_seen_point_stage_5}, length {np.size(index_seen_point_stage_5)}')
print(f'Stage 5:  Unseen point {index_unseen_point_stage_5}, length {np.size(index_unseen_point_stage_5)}\n\n')


# Sixth part moving horizontal, Y change, X remain constant from [244.223007, 549.171021], to [244.223007, 589.971008]
index_start_point_stage6 = np.argwhere(np.all(location_train[:, 0:2] == np.array([244.223007, 549.171021], dtype='float32'), axis=1)).squeeze()
print(f'Stage 6:  start location (X: {location_train[index_start_point_stage6,:].squeeze()[0]}  '
      f'Y: {location_train[index_start_point_stage6,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_start_point_stage6,:].squeeze()[2]})')

index_end_point_stage6 = np.argwhere(np.all(location_train[:, 0:2] == np.array([244.223007, 589.971008], dtype='float32'), axis=1)).squeeze()
print(f'Stage 3:  end location (X: {location_train[index_end_point_stage6,:].squeeze()[0]}  '
      f'Y: {location_train[index_end_point_stage6,:].squeeze()[1]}, '
      f'usrIndex: {location_train[index_end_point_stage6,:].squeeze()[2]})')

index_totalPoint_select_stage_6 = np.arange(index_start_point_stage6, index_end_point_stage6+1, 30)
print(f'Stage 6:  all point index ( X stable Y change): {index_totalPoint_select_stage_6}, length {np.size(index_totalPoint_select_stage_6)}')

indices = [i for i in range(0, len(index_totalPoint_select_stage_6), 5)]
indices.append(len(index_totalPoint_select_stage_6)-1)
index_seen_point_stage_6 = index_totalPoint_select_stage_6[indices]
index_unseen_point_stage_6 = np.setdiff1d(index_totalPoint_select_stage_6, index_seen_point_stage_6)
print(f'Stage 6:  Seen point {index_seen_point_stage_6}, length {np.size(index_seen_point_stage_6)}')
print(f'Stage 6:  Unseen point {index_unseen_point_stage_6}, length {np.size(index_unseen_point_stage_6)}\n\n')


index_seen_point_ALL = np.concatenate((index_seen_point_stage_1, index_seen_point_stage_2, index_seen_point_stage_3,
                                       index_seen_point_stage_4, index_seen_point_stage_5, index_seen_point_stage_6))

index_unseen_point_ALL = np.concatenate((index_unseen_point_stage_1, index_unseen_point_stage_2, index_unseen_point_stage_3,
                                         index_unseen_point_stage_4, index_unseen_point_stage_5, index_unseen_point_stage_6))

print(f'all Seen point index: {index_seen_point_ALL}, length {np.size(index_seen_point_ALL)}')
print(f'all Unseen point index: {index_unseen_point_ALL}, length {np.size(index_unseen_point_ALL)}')

# np.save('./data/mobility_test/index_seen_point_ALL.npy', index_seen_point_ALL)
# np.save('./data/mobility_test/index_unseen_point_ALL.npy', index_unseen_point_ALL)

seen_location = location_train[index_seen_point_ALL]
unseen_location = location_train[index_unseen_point_ALL]

