import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data extracted from the document
# data = {
#     "Configuration": ["No Augmentation", "VAE 100", "VAE 200", "VAE 400"],
#     # "Train Loss Space False Time True": [4.777685041780825, 4.765414944401494, 4.737614154815674, 4.732835398779975],
#     "Test Loss Space False Time True": [4.726081441949915, 4.754409648753978, 4.7525224685668945, 4.756516597889088],
#     # "Train Loss Space True Time False": [4.798109743330214, 4.750131395128038, 4.746011734008789, 4.741632797099926],
#     "Test Loss Space True Time False": [4.797850608825684, 4.79951016108195, 4.810379346211751, 4.8142398198445635],
#     # "Train Loss Space True Time True": [4.807686381869846, 4.756062825520833, 4.752306426012957, 4.740886105431451],
#     "Test Loss Space True Time True": [4.770403243877269, 4.7780093705212625, 4.7857785048308195, 4.786563802648473]
# }
#
# # Creating DataFrame
# df = pd.DataFrame(data)
# # Plotting Train Loss with Line Plot
# # plt.figure(figsize=(14, 7))
# #
# # # Line plot for train loss
# # for column in df.columns[1::2]:  # Selecting only train loss columns
# #     plt.plot(df["Configuration"], df[column], marker='o', label=column)
# #
# # plt.title('Train Loss for Different Configurations')
# # plt.xlabel('Configuration')
# # plt.ylabel('Train Loss')
# # plt.legend()
# # plt.grid(True)
# # plt.xticks(rotation=45)
# #
# # # Show the line plot
# # plt.tight_layout()
# # plt.show()
#
# # Plotting Test Loss with Bar Plot
# # fig, ax = plt.subplots(figsize=(14, 7))
# #
# # # Bar plot for test loss
# # for i, column in enumerate(df.columns[2::2], start=1):  # Selecting only test loss columns
# #     ax.bar(df["Configuration"], df[column], width=0.2, label=column, align='center', alpha=0.7, position=i)
# #
# # plt.title('Test Loss for Different Configurations')
# # plt.xlabel('Configuration')
# # plt.ylabel('Test Loss')
# # plt.legend()
# # plt.grid(True)
# # plt.xticks(rotation=45)
# #
# # # Show the bar plot
# # plt.tight_layout()
# # plt.show()
#
#
# # Creating separate dataframes for each condition to facilitate plotting
# conditions = ['Space False Time True', 'Space True Time False', 'Space True Time True']
# condition_data = {cond: df.filter(like=cond) for cond in conditions}
#
# # Define the limits for y-axis to better visualize the differences
# y_min = 4.726081441949915 - 0.01  # minimum value in the data minus a small number for padding
# y_max = 4.8142398198445635 + 0.01  # maximum value in the data plus a small number for padding
#
# color_name = 'summer'
# select1 = (130, 10)
# colors = plt.get_cmap(color_name)(select1)
#
# # Plotting each condition
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 7), dpi=300)
#
# # Loop through each condition and plot on a separate subplot
# for ax, (cond, data) in zip(axes, condition_data.items()):
#     # Plotting Train Loss with Line Plot
#     # data.plot(kind='line', marker='o', ax=ax, ylim=(y_min, y_max))
#     # Plotting Test Loss with Bar Plot overlaid
#     data.plot(kind='bar', alpha=0.7, ax=ax, width=0.5, ylim=(y_min, y_max), color=colors)
#     ax.plot(np.arange(data.shape[0])-0.5/(2*data.shape[1]), data[data.columns[0]], marker='o', color=colors[0])
#     ax.plot(np.arange(data.shape[0])+0.5/(2*data.shape[1]), data[data.columns[1]], marker='o', color=colors[1])
#
#     ax.set_title(f'Train and Test Loss: {cond}')
#     # ax.set_xlabel('Configuration')
#     ax.set_ylabel('Absolute Loss')
#     ax.legend(["Train Date Rate", "Test Date Rate"])
#     ax.grid(True)
#     ax.set_xticklabels(df["Configuration"], rotation=45)
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.savefig('./data_aug_result.png')
# plt.show()

data = {
    "Configuration": ["Only Time-Domain Test", "Only Spatial-Domain Test", "Joint Time-Spacial-Domain Test"],
    "No Augmentation": [4.726081441949915, 4.797850608825684, 4.770403243877269],
    "VAE 100": [4.751733762246591, 4.79951016108195, 4.7780093705212625],
    "VAE 200": [4.7525224685668945, 4.810379346211751, 4.7857785048308195],
    "VAE 400": [4.756516597889088, 4.8142398198445635, 4.786563802648473]
}
# 如何根据data字典结构判断要画柱状图的列数？   1. 通过data.keys()得到所有的列名，然后除去第一列，剩下的就是要画柱状图的列名
# 2. 通过data.values()得到所有的列值，然后除去第一列，剩下的就是要画柱状图的列值
# 3. 通过data.items()得到所有的列名和列值，然后除去第一列，剩下的就是要画柱状图的列名和列值

# Creating DataFrame
df = pd.DataFrame(data)

x_zhexian_in_colomn_0 = [0 - 3 * 0.5 / (2 * (df.shape[1] - 1)), 0 - 1 * 0.5 / (2 * (df.shape[1] - 1)),
                         0 + 1 * 0.5 / (2 * (df.shape[1] - 1)), 0 + 3 * 0.5 / (2 * (df.shape[1] - 1))]  # x_zhexian_in_colomn_0表示第一列的x轴坐标
x_zhexian_in_colomn_1 = [1 - 3 * 0.5 / (2 * (df.shape[1] - 1)), 1 - 1 * 0.5 / (2 * (df.shape[1] - 1)),
                         1 + 1 * 0.5 / (2 * (df.shape[1] - 1)), 1 + 3 * 0.5 / (2 * (df.shape[1] - 1))]  # x_zhexian_in_colomn_1表示第二列的x轴坐标
x_zhexian_in_colomn_2 = [2 - 3 * 0.5 / (2 * (df.shape[1] - 1)), 2 - 1 * 0.5 / (2 * (df.shape[1] - 1)),
                         2 + 1 * 0.5 / (2 * (df.shape[1] - 1)), 2 + 3 * 0.5 / (2 * (df.shape[1] - 1))]  # x_zhexian_in_colomn_2表示第三列的x轴坐标

# Define the limits for y-axis to better visualize the differences
y_min = 4.726081441949915 - 0.01  # minimum value in the data minus a small number for padding
y_max = 4.8142398198445635 + 0.01  # maximum value in the data plus a small number for padding

color_name = 'summer'
select1 = (200, 150, 70, 10)
colors = plt.get_cmap(color_name)(select1)

color_name_zhexian = 'summer'
select2 = [100, 100, 100]
colors_zhexian = plt.get_cmap(color_name_zhexian)(select2)

# Adjusting the grid lines to make major grid lines thicker
sns.set(style="whitegrid", palette="muted", context='paper')   #代表设置背景为白色网格，调色板为muted，上下文为paper
plt.figure(figsize=(12, 8))
df.plot(kind='bar', alpha=0.7, width=0.5, ylim=(y_min, y_max), color=colors)
plt.legend(["No Augmentation", "Augmentation: 100 data(125%)", "Augmentation: 200 data(250%)", "Augmentation: 400 data(500%)"], loc='upper left')

plt.plot(x_zhexian_in_colomn_0, df.iloc[0].tolist()[1:], marker='o', color=colors_zhexian[0])   # df.iloc[0].tolist()[1:]表示取第一行的第二列到最后一列的值
plt.plot(x_zhexian_in_colomn_1, df.iloc[1].tolist()[1:], marker='o', color=colors_zhexian[1])   # df.iloc[1].tolist()[1:]表示取第二行的第二列到最后一列的值
plt.plot(x_zhexian_in_colomn_2, df.iloc[2].tolist()[1:], marker='o', color=colors_zhexian[2])   # df.iloc[2].tolist()[1:]表示取第三行的第二列到最后一列的值

locs, _ = plt.xticks()
# plt.title('Average Spectral Efficiency with VAE-Based Channel Augmentation')
plt.ylabel('Average Spectral Efficiency (bps/Hz)')
plt.xlabel('Data Augmentation Test Configuration')
ax = plt.gca()  # Get current axis
# Enable and customize grid
ax.grid(which='major', linestyle='-', linewidth='0.5')  # Thicker lines for major grid

# Restoring the complete axes and showing the plot
sns.despine(offset={'left': 10, 'bottom': 2}, trim=False)   # offset表示坐标轴距离图像的距离，trim表示是否裁剪坐标轴
plt.xticks(locs, df["Configuration"], rotation=5)
plt.tight_layout()
# plt.savefig('./fig/space_time_dataaug_fig/data_aug_test.pdf')
plt.savefig('./fig/space_time_dataaug_fig/data_aug.pdf')
plt.show()

