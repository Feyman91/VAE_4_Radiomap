import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def draw_cdf():
    # File paths for the four datasets
    file_paths = {
        "Real Channels SVD Precoding": './fig/cdf/data_rate_best_svd_eachpoint.npy',
        "VAE-Gen Channels SVD Precoding": './fig/cdf/data_rate_precoder_VAE.npy'
        # "Oversampled-Codebook Precoding": './fig/cdf/data_rate_oversampled_codebook_mode.npy'
    }
    # Loading the data from each file
    datasets = {label: np.load(path) for label, path in file_paths.items()}
    # Adjusting the grid lines to make major grid lines thicker
    sns.set(style="whitegrid", palette="muted", context='poster')

    # Create a color palette
    palette = sns.color_palette("hls", 4)[0:]  # HUSL color palette

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Plotting the main CDF curves
    for (label, data), color in zip(datasets.items(), palette):
        data_sorted = np.sort(data)
        cumulative_prob = np.arange(1, len(data) + 1) / len(data)
        plt.plot(data_sorted, cumulative_prob, label=label, alpha=0.8, color=color, lw=5)
    # Set titles and labels
    # plt.title(f'Comparison of Datarate CDF with Different channel dataset', fontsize=15)
    plt.xlabel('Spectral Efficiency (Bps/Hz)')
    plt.ylabel('Cumulative Probability')
    plt.legend()

    # Adjusting x-axis major and minor ticks
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))  # Major x-axis ticks every 0.5 units
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Minor x-axis ticks every 0.1 units

    # Enable and customize grid
    ax.grid(which='major', linestyle='-', linewidth='0.5')  # Thicker lines for major grid
    ax.grid(which='minor', linestyle=':', linewidth='0.3')  # Thinner lines for minor grid

    # Restoring the complete axes and showing the plot
    sns.despine(offset={'bottom': 5, 'left': 10}, trim=False)
    plt.tight_layout()
    # plt.savefig(f'./fig/cdf/Datarate_CDF_with_Different_channel_dataset.svg')
    plt.show()

def draw_rate_zhuzhuangtu():
    values = [4.7943623079338, 4.705306581917615]
    labels = ["Real Channels SVD Precoding", "VAE-Gen Channels SVD Precoding"]
    colors = sns.color_palette("hls", 4)[0:]
    y_min = 2 - 0.01  # minimum value in the data minus a small number for padding
    y_max = 5 + 0.01  # maximum value in the data plus a small number for padding

    sns.set(style="whitegrid", palette="muted", context='poster')
    # 设置绘图尺寸
    plt.figure(figsize=(12, 8))

    # 绘制柱状图
    bars = plt.bar(labels, values, color=colors, alpha=0.8, width=0.25)
    plt.ylim(y_min, y_max)

    # 设置标签
    # plt.title('SVD Precoding Values for Real and VAE-Generated Channels')
    plt.xlabel('Channel Precoding Methods')
    plt.ylabel('Average Spectral Efficiency (Bps/Hz)')

    # 在每个柱状图上添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.4f}',
                 ha='center', va='bottom')

    ax = plt.gca()  # Get current axis
    ax.tick_params(axis='x', rotation=5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))  # Set major x-axis ticks every 0.5 units
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))  # Keep minor x-axis ticks every 0.1 units
    # Enable and customize grid
    ax.grid(which='major', linestyle='-', linewidth='0.7')  # Thicker lines for major grid
    ax.grid(which='minor', linestyle=':', linewidth='0.3')  # Thinner lines for minor grid

    # Restoring the complete axes and showing the plot
    sns.despine(offset={'bottom': 1, 'left': 7}, trim=False)
    plt.tight_layout()
    # plt.savefig(f'./fig/cdf/Datarate_zhuzhuangtu_compare.svg')
    plt.show()

if __name__ == '__main__':
    draw_cdf()
    # draw_rate_zhuzhuangtu()




