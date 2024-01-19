import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# 加载 MATLAB 文件
file_path = 'F:/Ph.D/Magazine/vae/matlab/precoder_test.mat'
mat_data = scipy.io.loadmat(file_path)

# 提取预编码器数据
precoder_svd = mat_data['precoder_svd']
precoder_vae = mat_data['precoder_vae']
precoder_codebook = mat_data['precoder_codebook']

# 重新生成Hset_adjusted矩阵
# 使用N个天线元素生成天线方向矩阵，确保其与预编码器矩阵的行数匹配
N = 16  # 假设天线数
idx = np.arange(N)
angle_set_adjusted = np.linspace(0, np.pi, 1800)  # 与预编码器矩阵的行数相匹配的角度集合
Hset_adjusted = np.exp(1j * np.pi * idx.reshape(-1, 1) * np.cos(angle_set_adjusted))
# 重新定义绘制天线方向图的函数
palette_svd = sns.color_palette("prism", 4) # Set1, prism, Paired
palette_vae = sns.color_palette("Set1", 4)[2:]  #
# 上面两行代码是设置调色板，可以自己定义，也可以使用seaborn自带的调色板，这里使用的是seaborn自带的调色板
# HUSL color palette
def plot_angel_spectra(spectra, ax):
    for (label, spectrum), color in zip(spectra.items(), palette_svd):
        ax.plot(angle_set_adjusted * 180 / np.pi, spectrum, label=label, color=palette_svd[0])
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Magnitude')
    plt.legend(fontsize=12)

    ax.set_ylim([0, 1])
    ax.set_xlim([0, 180])
    ax.set_xticks(np.arange(0, 181, 20))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))  # Set major x-axis ticks every 0.5 units
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))  # Keep minor x-axis ticks every 0.1 units
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # Set major x-axis ticks every 0.5 units
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Keep minor x-axis ticks every 0.1 units
    # Enable and customize grid
    ax.grid(which='major', linestyle='-', linewidth='0.4')  # Thicker lines for major grid
    ax.grid(which='minor', linestyle=':', linewidth='0.2')  # Thinner lines for minor grid
    sns.despine(offset={'left': 0, 'bottom': 3}, right=False, trim=False)   # 这行代码是去除坐标轴上的线，offset表示坐标轴距离图像的距离，trim表示是否裁剪坐标轴，right表示是否裁剪右边的坐标轴



def compute_angel_spectrum_corrected(precoder, user_idx):
    angel_spectrum = Hset_adjusted.T @ precoder[user_idx - 1, :]
    return np.abs(angel_spectrum) / np.max(np.abs(angel_spectrum))

# 用户索引
# user = [560, 1324, 1700, 2650]
user = [560]
angel_spectrum_svd_corrected = [compute_angel_spectrum_corrected(precoder_svd, u) for u in user]
angel_spectrum_vae_corrected = [compute_angel_spectrum_corrected(precoder_vae, u) for u in user]
# angel_spectrum_codebook_corrected = [compute_angel_spectrum_corrected(precoder_codebook, u) for u in user]
angel_spectrum_svd_corrected_test = {f'Channel {idx+1}': angel_spectrum_svd_corrected[idx] for idx in range(len(angel_spectrum_svd_corrected))}
angel_spectrum_vae_corrected_test = {f'Channel {idx+1}': angel_spectrum_vae_corrected[idx] for idx in range(len(angel_spectrum_svd_corrected))}
# angel_spectrum_codebook_corrected_test = {f'Channel {idx+1}': angel_spectrum_codebook_corrected[idx] for idx in range(len(angel_spectrum_svd_corrected))}


dicts_list = [angel_spectrum_svd_corrected_test, angel_spectrum_vae_corrected_test]

# 重新计算天线方向图

# 重新绘制和美化图形
sns.set(style="whitegrid", palette="muted", context='talk')
plt.figure()
axe1 = plt.gca()
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# plot_angel_spectra(angel_spectrum_svd_corrected_test, axes[0])
# plot_angel_spectra(angel_spectrum_vae_corrected_test, axes[1])
# plot_angel_spectra(angel_spectrum_codebook_corrected_test, axes[2])

plot_angel_spectra(angel_spectrum_svd_corrected_test, axe1)

legend = axe1.get_legend()
legend.remove()
# vae  svd  codebook
plt.tight_layout()
# plt.savefig(f'./fig/vae_precoder_tianxian_polar_drawpic/new/svd_precoder_cmp_user0.svg')
plt.show()

