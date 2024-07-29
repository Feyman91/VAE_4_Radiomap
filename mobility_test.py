# import tensorflow as tf
# from tensorflow import keras
from model import ANN_TypeI
from collections import defaultdict
from utils import *
import torch.utils.data as Data
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

torch.backends.cudnn.deterministic = True  # 这一行设置为True的话，每次运行网络的时候相同输入的输出是固定的
torch.backends.cudnn.benchmark = False      # 这一行设置为False的话，每次运行网络的时候相同输入的输出是固定的
g = torch.Generator()
g.manual_seed(2023)
np.random.seed(2023)
N_f = 1  # subcarrier number
N_tx = 16  # Tx antenna number
N_rx = 4  # Rx antenna number
N_point = 3780
N_history = 100
N_org = 60
N_eval_time = 40

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# radio_map_file_name_loss = f'./networks/space{is_space_eval}_time{is_time_eval}_joint_prediction_org{N_org}_aug{is_aug}{N_aug}/loss'
# check_directory(radio_map_file_name)


def dataload():
    f = scio.loadmat('./data/train_new.mat')
    data = f['train']
    location_data = data[0, 0]['loc']
    channel_data = data[0, 0]['CSI']
    location_train = location_data[::100, ...]
    location_train[:, 2] = np.array(list(range(0, 3780)))
    index_seen_point_ALL = np.load('./data/mobility_test/index_seen_point_ALL.npy')
    index_Unseen_point_ALL = np.load('./data/mobility_test/index_Unseen_point_ALL.npy')
    precoder_svd = np.load('./data/precoder_svd_eachpoint.npy')
    print(f'<<<<origin channel data shape: {channel_data.shape}>>>>')
    print(f'<<<<location data shape: {location_train.shape}>>>>')
    print(f'<<<<svd precoder size: {precoder_svd.shape}>>>>')
    print(f'<<<<seen location index size: {index_seen_point_ALL.shape}>>>>')
    print(f'<<<<Unseen location index size: {index_Unseen_point_ALL.shape}>>>>\n')

    return channel_data, location_train, index_seen_point_ALL, index_Unseen_point_ALL, precoder_svd


class SeenDataSetForMobilityTest(Data.Dataset):
    def __init__(self, location_train, channel_data, index_seen_point_ALL, precoder_svd):
        super(SeenDataSetForMobilityTest, self).__init__()
        self.location = location_train  # 根据输入index来确定
        self.index_seen_point_ALL = index_seen_point_ALL
        self.location_eval_on_seenData = location_train[index_seen_point_ALL]
        self.precoder_svd = precoder_svd[index_seen_point_ALL]

        channel_org_temp = channel_data.reshape(N_point, N_history, N_rx, N_tx, N_f, 2)
        channel_eval_on_seenData_temp = channel_org_temp[index_seen_point_ALL, ...]
        channel_eval_on_seenData = channel_eval_on_seenData_temp[:, N_org:N_history, ...].reshape(-1, N_rx, N_tx, N_f,
                                                                                                  2)  # 这步对原始数据集进行时间的筛选
        print(f"------eval channel shape on seenData{channel_eval_on_seenData.shape}------")

        N_used = N_org
        self.channel_eval_on_seenData = channel_eval_on_seenData
        self.N_used = N_used

    def __len__(self):
        return self.location_eval_on_seenData.shape[0]

    def __getitem__(self, idx):
        location_current = torch.FloatTensor(self.location_eval_on_seenData[idx, 0:2])
        precoder_current = torch.FloatTensor(self.precoder_svd[idx, ...])
        channel_eval = torch.FloatTensor(self.channel_eval_on_seenData[idx * N_eval_time:(idx + 1) * N_eval_time, ...])
        return location_current, channel_eval, precoder_current


class UnseenDataSetForMobilityTest(Data.Dataset):
    def __init__(self, location_train, channel_data, index_Unseen_point_ALL, precoder_svd):
        super(UnseenDataSetForMobilityTest, self).__init__()
        self.location = location_train  # 根据输入index来确定
        self.index_unseen_point_ALL = index_Unseen_point_ALL
        self.location_eval_on_unseenData = location_train[index_Unseen_point_ALL]
        self.precoder_svd = precoder_svd[index_Unseen_point_ALL]

        channel_org_temp = channel_data.reshape(N_point, N_history, N_rx, N_tx, N_f, 2)
        channel_eval_on_unseenData_temp = channel_org_temp[index_Unseen_point_ALL, ...]
        channel_eval_on_unseenData = channel_eval_on_unseenData_temp[:, N_org:N_history, ...].reshape(-1, N_rx, N_tx,
                                                                                                      N_f,
                                                                                                      2)  # 这步对原始数据集进行时间的筛选
        print(f"------eval channel shape on unseenData{channel_eval_on_unseenData.shape}------")

        N_used = N_org
        self.channel_eval_on_unseenData = channel_eval_on_unseenData
        self.N_used = N_used

    def __len__(self):
        return self.location_eval_on_unseenData.shape[0]

    def __getitem__(self, idx):
        location_current = torch.FloatTensor(self.location_eval_on_unseenData[idx, 0:2])
        precoder_current = torch.FloatTensor(self.precoder_svd[idx, ...])
        channel_eval = torch.FloatTensor(
            self.channel_eval_on_unseenData[idx * N_eval_time:(idx + 1) * N_eval_time, ...])
        return location_current, channel_eval, precoder_current


class AllDataSetForMobilityTest(Data.Dataset):
    def __init__(self, location_train, channel_data, precoder_svd, index_seen_point_ALL, index_Unseen_point_ALL):
        super(AllDataSetForMobilityTest, self).__init__()
        self.location = location_train  # 根据输入index来确定
        index_all = np.sort(np.concatenate((index_seen_point_ALL, index_Unseen_point_ALL)))
        self.index_point_ALL = index_all
        self.location_eval_on_allpoint = location_train[index_all]
        self.precoder_svd = precoder_svd[index_all]

        channel_org_temp = channel_data.reshape(N_point, N_history, N_rx, N_tx, N_f, 2)
        channel_eval_on_allpoint_temp = channel_org_temp[index_all, ...]
        channel_eval_on_allpoint = channel_eval_on_allpoint_temp[:, N_org:N_history, ...].reshape(-1, N_rx, N_tx,
                                                                                                  N_f,
                                                                                                  2)  # 这步对原始数据集进行时间的筛选
        print(f"------eval channel shape on all data point{channel_eval_on_allpoint.shape}------")

        N_used = N_org
        self.channel_eval_on_allpoint = channel_eval_on_allpoint
        self.N_used = N_used

    def __len__(self):
        return self.location_eval_on_allpoint.shape[0]

    def __getitem__(self, idx):
        location_current = torch.FloatTensor(self.location_eval_on_allpoint[idx, 0:2])
        precoder_current = torch.FloatTensor(self.precoder_svd[idx, ...])
        channel_eval = torch.FloatTensor(
            self.channel_eval_on_allpoint[idx * N_eval_time:(idx + 1) * N_eval_time, ...])
        return location_current, channel_eval, precoder_current


def computing_datarate(is_aug, N_aug, channel_data, location_train, index_seen_point_ALL, index_Unseen_point_ALL,
                       precoder_svd):
    batch_size = 1
    data_rate_seen_collection = defaultdict(list)
    data_rate_unseen_collection = defaultdict(list)
    model = ANN_TypeI().to(device)  # 时间和频率上都取众数的网络
    if is_aug:  # 是否使用增强数据训练的模型进行测试
        print(f'\n<<<<loading model trained with {N_aug} augment data>>>>')
        if N_aug == 100:
            model.load_state_dict(torch.load(
                "./networks/verify_for_mobility/space_verify_best_100aug/epoch199_trainloss-4.750131395128038_verifyloss-4.79951016108195.pt"))
        elif N_aug == 200:
            model.load_state_dict(torch.load(
                "./networks/verify_for_mobility/space_verify_best_200aug/epoch172_trainloss-4.746011734008789_verifyloss-4.810379346211751.pt"))
        elif N_aug == 400:
            model.load_state_dict(torch.load(
                "./networks/verify_for_mobility/space_verify_best_400aug/epoch187_trainloss-4.741632797099926_verifyloss-4.8142398198445635.pt"))
        else:
            print("N_aug is wrong !!!!!!!")
            exit()
    else:
        print(f'\n<<<<loading model trained without augment>>>>')
        model.load_state_dict(torch.load(
            "./networks/verify_for_mobility/space_verify_best_noaug/epoch196_trainloss-4.798109743330214_verifyloss-4.797850608825684.pt"))
    loss_fn = DataRateLoss()
    dataset_mobilityTest_seen = SeenDataSetForMobilityTest(location_train, channel_data, index_seen_point_ALL,
                                                           precoder_svd)
    dataset_mobilityTest_unseen = UnseenDataSetForMobilityTest(location_train, channel_data, index_Unseen_point_ALL,
                                                               precoder_svd)
    dataset_iter_seen = Data.DataLoader(dataset_mobilityTest_seen, batch_size=batch_size, shuffle=True)
    dataset_iter_unseen = Data.DataLoader(dataset_mobilityTest_unseen, batch_size=batch_size, shuffle=True)

    datarate_seen, min_rate_seen, max_rate_seen = verify_step_mobility_test_seendata(dataset_iter_seen, model, loss_fn,
                                                                                     N_eval_time)
    datarate_unseen, min_rate_unseen, max_rate_unseen = verify_step_mobility_test_Unseendata(dataset_iter_unseen, model,
                                                                                             loss_fn, N_eval_time)

    data_rate_seen_collection['datarate_seen'] = datarate_seen
    data_rate_seen_collection['min_rate_seen'] = min_rate_seen
    data_rate_seen_collection['max_rate_seen'] = max_rate_seen

    data_rate_unseen_collection['datarate_unseen'] = datarate_unseen
    data_rate_unseen_collection['min_rate_unseen'] = min_rate_unseen
    data_rate_unseen_collection['max_rate_unseen'] = max_rate_unseen

    print("Done!\n")

    return data_rate_seen_collection, data_rate_unseen_collection


def computing_datarate_svd_precoding(channel_data, location_train, index_seen_point_ALL, index_Unseen_point_ALL,
                                     precoder_svd):
    batch_size = 1
    data_rate_collection = defaultdict(list)
    data_rate_collection_seensvd = defaultdict(list)
    data_rate_collection_unseensvd = defaultdict(list)
    print(f'\n<<<<computing svd precoder datarate for mobility baseline>>>>')
    dataset_mobilityTest_allpoint = AllDataSetForMobilityTest(location_train, channel_data, precoder_svd,
                                                              index_seen_point_ALL, index_Unseen_point_ALL)
    dataset_mobilityTest_seenpointSvd = SeenDataSetForMobilityTest(location_train, channel_data, index_seen_point_ALL,
                                                                   precoder_svd)
    dataset_mobilityTest_unseenpointSvd = UnseenDataSetForMobilityTest(location_train, channel_data,
                                                                       index_Unseen_point_ALL, precoder_svd)
    dataset_iter_allpoint = Data.DataLoader(dataset_mobilityTest_allpoint, batch_size=batch_size, shuffle=True)
    dataset_iter_seenpointSvd = Data.DataLoader(dataset_mobilityTest_seenpointSvd, batch_size=batch_size, shuffle=True)
    dataset_iter_unseenpointSvd = Data.DataLoader(dataset_mobilityTest_unseenpointSvd, batch_size=batch_size,
                                                  shuffle=True)

    loss_fn = DataRateLoss()
    datarate_svd, min_rate_svd, max_rate_svd = verify_step_mobility_test_svdPrecoder_allpoint(dataset_iter_allpoint,
                                                                                              loss_fn, N_eval_time)
    datarate_seen_svd, min_rate_seen_svd, max_rate_seen_svd = verify_step_mobility_test_svdPrecoder_allpoint(
        dataset_iter_seenpointSvd, loss_fn, N_eval_time, )
    datarate_unseen_svd, min_rate_unseen_svd, max_rate_unseen_svd = verify_step_mobility_test_svdPrecoder_allpoint(
        dataset_iter_unseenpointSvd, loss_fn, N_eval_time)

    data_rate_collection['datarate_seen'] = datarate_svd
    data_rate_collection['min_rate_seen'] = min_rate_svd
    data_rate_collection['max_rate_seen'] = max_rate_svd
    data_rate_collection_seensvd['datarate_seen'] = datarate_seen_svd
    data_rate_collection_seensvd['min_rate_seen'] = min_rate_seen_svd
    data_rate_collection_seensvd['max_rate_seen'] = max_rate_seen_svd
    data_rate_collection_unseensvd['datarate_seen'] = datarate_unseen_svd
    data_rate_collection_unseensvd['min_rate_seen'] = min_rate_unseen_svd
    data_rate_collection_unseensvd['max_rate_seen'] = max_rate_unseen_svd
    print(f"all point svd datarate: {datarate_svd:>8f}")
    print(f"seen point svd datarate: {datarate_seen_svd:>8f}")
    print(f"unseen point svd datarate: {datarate_unseen_svd:>8f}")
    print("Done!\n")

    return data_rate_collection, data_rate_collection_seensvd, data_rate_collection_unseensvd


def main():
    print("yhnb")
    datarate_seendata_withDiff_test = []
    datarate_Useendata_withDiff_test = []
    datarate_seen_min_withDiff_test = []
    datarate_seen_max_withDiff_test = []
    datarate_unseen_min_withDiff_test = []
    datarate_unseen_max_withDiff_test = []
    channel_data, location_train, index_seen_point_ALL, index_Unseen_point_ALL, precoder_svd = dataload()
    data_rate_svd_collection, data_rate_collection_seensvd, data_rate_collection_unseensvd = computing_datarate_svd_precoding(
        channel_data, location_train, index_seen_point_ALL, index_Unseen_point_ALL, precoder_svd)
    for is_aug in [False, True]:
        if is_aug:
            for N_aug in [100, 200, 400]:
                print(f'------is_aug: {is_aug}, N_aug: {N_aug}-------')
                data_rate_seen_collection, data_rate_unseen_collection = computing_datarate(is_aug, N_aug, channel_data,
                                                                                            location_train,
                                                                                            index_seen_point_ALL,
                                                                                            index_Unseen_point_ALL,
                                                                                            precoder_svd)
                # np.save(
                #     f'./networks/verify_for_mobility/space_verify_best_{N_aug}aug/datarate_seen_unseen_{is_aug}_{N_aug}',
                #     [datarate_seen, datarate_unseen])
                datarate_seendata_withDiff_test.append(data_rate_seen_collection['datarate_seen'])
                datarate_seen_min_withDiff_test.append(data_rate_seen_collection['min_rate_seen'])
                datarate_seen_max_withDiff_test.append(data_rate_seen_collection['max_rate_seen'])
                datarate_Useendata_withDiff_test.append(data_rate_unseen_collection['datarate_unseen'])
                datarate_unseen_min_withDiff_test.append(data_rate_unseen_collection['min_rate_unseen'])
                datarate_unseen_max_withDiff_test.append(data_rate_unseen_collection['max_rate_unseen'])
        else:
            N_aug = 0
            print(f'------is_aug: {is_aug}, N_aug: Nah-------')
            data_rate_seen_collection, data_rate_unseen_collection = computing_datarate(is_aug, N_aug, channel_data,
                                                                                        location_train,
                                                                                        index_seen_point_ALL,
                                                                                        index_Unseen_point_ALL,
                                                                                        precoder_svd)
            # np.save(
            #     f'./networks/verify_for_mobility/space_verify_best_{N_aug}aug/datarate_seen_unseen_{is_aug}_{N_aug}',
            #     [datarate_seen, datarate_unseen])
            datarate_seendata_withDiff_test.append(data_rate_seen_collection['datarate_seen'])
            datarate_seen_min_withDiff_test.append(data_rate_seen_collection['min_rate_seen'])
            datarate_seen_max_withDiff_test.append(data_rate_seen_collection['max_rate_seen'])
            datarate_Useendata_withDiff_test.append(data_rate_unseen_collection['datarate_unseen'])
            datarate_unseen_min_withDiff_test.append(data_rate_unseen_collection['min_rate_unseen'])
            datarate_unseen_max_withDiff_test.append(data_rate_unseen_collection['max_rate_unseen'])

    plt.plot(datarate_seendata_withDiff_test, label='seen_data')
    plt.plot(datarate_Useendata_withDiff_test, label='Unseen_data')
    plt.legend()
    # plt.savefig(f'./networks/verify_for_mobility/space_verify_best_{N_aug}aug/datarate_fig.png')
    plt.show()
    # np.save(datarate_seen + f'/losses_train_verify_epoch{i}', losses)
    print(f'------datarate_seendata_withDiff_test: {datarate_seendata_withDiff_test}------')
    print(f'------datarate_seen_min_withDiff_test: {datarate_seen_min_withDiff_test}------')
    print(f'------datarate_seen_max_withDiff_test: {datarate_seen_max_withDiff_test}------')
    print(f'\n------datarate_Useendata_withDiff_test: {datarate_Useendata_withDiff_test}------')
    print(f'------datarate_unseen_min_withDiff_test: {datarate_unseen_min_withDiff_test}------')
    print(f'------datarate_unseen_max_withDiff_test: {datarate_unseen_max_withDiff_test}------')


def draw():
    # datarate_seendata_withDiff_test = np.array([5.014226913452148, 5.027305603027344, 5.037172317504883, 5.049903869628906], dtype=np.float32)
    # datarate_Useendata_withDiff_test = np.array([4.943161487579346, 4.957836627960205, 4.96189022064209, 4.965973377227783], dtype=np.float32)
    data = {
        "Configuration": ["No Augmentation", "VAE 100", "VAE 200", "VAE 400"],
        "Trained location in dataset": [5.014226913452148, 5.027305603027344, 5.037172317504883, 5.049903869628906],
        "Unseen location due to mobility": [4.943161487579346, 4.957836627960205, 4.96189022064209, 4.965973377227783],
    }

    df = pd.DataFrame(data)
    y_min = 4.8
    y_max = 5.2
    sns.set(style="whitegrid", palette="muted", context='paper')  # 代表设置背景为白色网格，调色板为muted，上下文为paper
    plt.figure(figsize=(12, 8))
    df.plot(alpha=0.7, ylim=(y_min, y_max))
    plt.legend(["Trained location in dataset", "Unseen location due to mobility"], loc='upper left')
    plt.ylabel('Average Spectral Efficiency (bps/Hz)')
    plt.xlabel('Data Augmentation Test Configuration')
    locs, _ = plt.xticks()
    ax = plt.gca()  # Get current axis
    # Enable and customize grid
    ax.grid(which='major', linestyle='-', linewidth='0.5')  # Thicker lines for major grid

    # Restoring the complete axes and showing the plot
    sns.despine(offset={'left': 10, 'bottom': 2}, trim=False)  # offset表示坐标轴距离图像的距离，trim表示是否裁剪坐标轴
    plt.xticks(np.array([0.0, 1.0, 2.0, 3.0]), df["Configuration"], rotation=5)
    plt.tight_layout()
    # plt.plot(datarate_seendata_withDiff_test, label='seen_data')
    # plt.plot(datarate_Useendata_withDiff_test, labell='Unseen_data')
    plt.show()


def draw2():
    # 输入数据
    data = {
        "Configuration": ["No Augmentation", "VAE 100", "VAE 200", "VAE 400"],
        "Trained location in dataset": [5.014226913452148, 5.027305603027344, 5.037172317504883, 5.049903869628906],
        "Unseen location due to mobility": [4.943161487579346, 4.957836627960205, 4.96189022064209, 4.965973377227783],
    }
    df = pd.DataFrame(data)

    # 设置风格和调色板
    sns.set(style="whitegrid", palette="muted", context='paper')

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(df["Configuration"], df["Trained location in dataset"], marker='o', linestyle='-',
             label='Trained location in dataset', linewidth=2.5)
    plt.plot(df["Configuration"], df["Unseen location due to mobility"], marker='o', linestyle='-',
             label='Unseen location due to mobility', linewidth=2.5)

    # 设置图例
    plt.legend(loc='upper left', fontsize=12)

    # 设置坐标轴标签
    plt.ylabel('Average Spectral Efficiency (bps/Hz)', fontsize=14)
    plt.xlabel('Data Augmentation Test Configuration', fontsize=14)

    # 设置y轴范围
    plt.ylim(4.8, 5.2)

    # 启用和自定义网格
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey', alpha=0.7)
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.7)

    # 移除顶部和右侧边框
    sns.despine(offset={'left': 10, 'bottom': 2}, trim=False)

    # 设置x轴刻度
    plt.xticks(np.arange(len(df["Configuration"])), df["Configuration"], rotation=5, fontsize=12)

    # 设置y轴刻度
    plt.yticks(fontsize=12)

    # 调整布局以避免标签重叠
    plt.tight_layout()

    # 显示图表
    plt.show()


def draw3():
    # 输入数据
    data = {
        "Configuration": ["No Augmentation", "VAE 100", "VAE 200", "VAE 400"],
        "Trained location in dataset": [5.01422702299582, 5.027306086308247, 5.0371726203609155, 5.049903373460512],
        "Unseen location due to mobility": [4.94316153252711, 4.957836608417699, 4.961890087753046, 4.96597285544286],
    }
    df = pd.DataFrame(data)

    # 假设的最小值和最大值数据
    min_max_data = {
        "Trained location in dataset": {
            "min": [2.7401058673858643, 2.7852985858917236, 2.8337135314941406, 2.8559558391571045],
            "max": [6.431890964508057, 6.433403968811035, 6.449571132659912, 6.4506683349609375]
        },
        "Unseen location due to mobility": {
            "min": [2.8720667362213135, 2.837557315826416, 2.8150665760040283, 2.8605241775512695],
            "max": [6.433804988861084, 6.437413215637207, 6.449320316314697, 6.451153755187988]
        }
    }

    # 设置风格和调色板
    sns.set(style="whitegrid", palette="muted", context='paper')

    # 绘图
    plt.figure(figsize=(12, 8))
    config_indices = np.arange(len(df["Configuration"]))

    # 绘制平均值和误差条
    plt.errorbar(config_indices, df["Trained location in dataset"],
                 yerr=[
                     np.subtract(df["Trained location in dataset"], min_max_data["Trained location in dataset"]["min"]),
                     np.subtract(min_max_data["Trained location in dataset"]["max"],
                                 df["Trained location in dataset"])],
                 fmt='-o', label='Trained location in dataset', linewidth=2.5)

    plt.errorbar(config_indices, df["Unseen location due to mobility"],
                 yerr=[np.subtract(df["Unseen location due to mobility"],
                                   min_max_data["Unseen location due to mobility"]["min"]),
                       np.subtract(min_max_data["Unseen location due to mobility"]["max"],
                                   df["Unseen location due to mobility"])],
                 fmt='-o', label='Unseen location due to mobility', linewidth=2.5)

    # 设置图例
    plt.legend(loc='upper left', fontsize=12)

    # 设置坐标轴标签
    plt.ylabel('Average Spectral Efficiency (bps/Hz)', fontsize=14)
    plt.xlabel('Data Augmentation Test Configuration', fontsize=14)

    # 设置y轴范围
    plt.ylim(4.8, 5.2)

    # 启用和自定义网格
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey', alpha=0.7)
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.7)

    # 移除顶部和右侧边框
    sns.despine(offset={'left': 10, 'bottom': 2}, trim=False)

    # 设置x轴刻度
    plt.xticks(config_indices, df["Configuration"], rotation=5, fontsize=12)

    # 设置y轴刻度
    plt.yticks(fontsize=12)

    # 调整布局以避免标签重叠
    plt.tight_layout()

    # 显示图表
    plt.show()


def draw4():
    # 输入数据
    data = {
        "Configuration": ["No Augmentation", "VAE 100", "VAE 200", "VAE 400"],
        "Trained location in dataset": [5.014226913452148, 5.027305603027344, 5.037172317504883, 5.049903869628906],
        "Unseen location due to mobility": [4.943161487579346, 4.957836627960205, 4.96189022064209, 4.965973377227783],
    }
    df = pd.DataFrame(data)
    theory_best = 5.659810157691908
    # 假设的最小值和最大值数据
    min_max_data = {
        "Trained location in dataset": {
            "min": [2.7401058673858643, 2.7852985858917236, 2.8337135314941406, 2.8559558391571045],
            "max": [6.431890964508057, 6.433403968811035, 6.449571132659912, 6.4506683349609375]
        },
        "Unseen location due to mobility": {
            "min": [2.8720667362213135, 2.837557315826416, 2.8150665760040283, 2.8605241775512695],
            "max": [6.433804988861084, 6.437413215637207, 6.449320316314697, 6.451153755187988]
        }
    }

    # 设置风格和调色板
    sns.set(style="whitegrid", palette="muted", context='paper')

    # 绘图
    plt.figure(figsize=(12, 8))
    config_indices = np.arange(len(df["Configuration"]))

    # 绘制平均值和误差条
    plt.errorbar(config_indices, df["Trained location in dataset"],
                 yerr=[
                     np.subtract(df["Trained location in dataset"], min_max_data["Trained location in dataset"]["min"]),
                     np.subtract(min_max_data["Trained location in dataset"]["max"],
                                 df["Trained location in dataset"])],
                 fmt='-o', label='Trained location in dataset', linewidth=2.5, markersize=10, capsize=5, capthick=2)

    plt.errorbar(config_indices, df["Unseen location due to mobility"],
                 yerr=[np.subtract(df["Unseen location due to mobility"],
                                   min_max_data["Unseen location due to mobility"]["min"]),
                       np.subtract(min_max_data["Unseen location due to mobility"]["max"],
                                   df["Unseen location due to mobility"])],
                 fmt='-o', label='Unseen location due to mobility', linewidth=2.5, markersize=10, capsize=5, capthick=2)

    # 设置图例
    plt.legend(loc='upper left', fontsize=12)

    # 设置坐标轴标签
    plt.ylabel('Average Spectral Efficiency (bps/Hz)', fontsize=14)
    plt.xlabel('Data Augmentation Test Configuration', fontsize=14)

    # 设置y轴范围
    # plt.ylim(4.8, 5.2)

    # 启用和自定义网格
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey', alpha=0.7)
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.7)

    # 移除顶部和右侧边框
    sns.despine(offset={'left': 10, 'bottom': 2}, trim=False)

    # 设置x轴刻度
    plt.xticks(config_indices, df["Configuration"], rotation=5, fontsize=12)

    # 设置y轴刻度
    plt.yticks(fontsize=15)

    # 调整布局以避免标签重叠
    plt.tight_layout()

    # 显示图表
    plt.show()


def draw5():
    # 输入数据
    data = {
        "Configuration": ["No Augmentation", "VAE 100", "VAE 200", "VAE 400"],
        "Trained location in dataset": [5.014226913452148, 5.027305603027344, 5.037172317504883, 5.049903869628906],
        "Unseen location due to mobility": [4.943161487579346, 4.957836627960205, 4.96189022064209, 4.965973377227783],
    }
    df = pd.DataFrame(data)

    # 设置风格和调色板
    sns.set(style="whitegrid", palette="muted", context='paper')

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(df["Configuration"], df["Trained location in dataset"], marker='o', linestyle='-',
             label='Trained location in dataset', linewidth=2.5, markersize=10)
    plt.plot(df["Configuration"], df["Unseen location due to mobility"], marker='o', linestyle='-',
             label='Unseen location due to mobility', linewidth=2.5, markersize=10)

    # 绘制SVD预编码的理论最高值
    svd_precoder_datarate_seenpoint = 5.034674934438757
    svd_precoder_datarate_unseenpoint = 4.9613323485265015
    plt.axhline(y=svd_precoder_datarate_seenpoint, color='b', linestyle='--', linewidth=2,
                label=' SVD precoding Maximum on seen point')
    plt.axhline(y=svd_precoder_datarate_unseenpoint, color='g', linestyle='--', linewidth=2,
                label='SVD Theoretical Maximum on unseen point')

    # 设置图例
    plt.legend(loc='upper left', fontsize=12)

    # 设置坐标轴标签
    plt.ylabel('Average Spectral Efficiency (bps/Hz)', fontsize=14)
    plt.xlabel('Data Augmentation Test Configuration', fontsize=14)

    # 设置y轴范围
    plt.ylim(4.89, 5.1)

    # 启用和自定义网格
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey', alpha=0.7)
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey', alpha=0.7)

    # 移除顶部和右侧边框
    sns.despine(offset={'left': 10, 'bottom': 2}, trim=False)

    # 设置x轴刻度
    plt.xticks(np.arange(len(df["Configuration"])), df["Configuration"], rotation=5, fontsize=12)

    # 设置y轴刻度
    plt.yticks(fontsize=12)

    # 调整布局以避免标签重叠
    plt.tight_layout()

    # 显示图表
    plt.show()


def draw6():
    data = {
        "Configuration": ["Static Scene Test", " UE Mobility Test"],
        "No Augmentation": [5.014226913452148, 4.943161487579346],
        "VAE 100": [5.027305603027344, 4.957836627960205],
        "VAE 200": [5.0334172317504883, 4.96189022064209],
        "VAE 400": [5.049903869628906, 4.965973377227783]
    }

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Define the limits for y-axis to better visualize the differences
    y_min = 4.891
    y_max = 5.112

    # Define colors for the bars
    color_name = 'Blues'
    select1 = (100, 150, 200, 250)
    colors = plt.get_cmap(color_name)(select1)

    # Set plot style
    sns.set(style="whitegrid", palette="muted", context='paper')

    # Create figure and axes
    plt.figure(figsize=(6, 4))

    # Create the bar chart
    bar_width = 0.1
    index = range(len(df['Configuration']))

    plt.bar([i for i in index], df['No Augmentation'], bar_width, label='No Augmentation', color=colors[0], alpha=0.8)
    plt.bar([i + bar_width for i in index], df['VAE 100'], bar_width, label='Augmentation: 100 data(125%)', color=colors[1], alpha=0.8)
    plt.bar([i + 2 * bar_width for i in index], df['VAE 200'], bar_width, label='Augmentation: 200 data(250%)', color=colors[2], alpha=0.8)
    plt.bar([i + 3 * bar_width for i in index], df['VAE 400'], bar_width, label='Augmentation: 400 data(500%)', color=colors[3], alpha=0.8)
    svd_precoder_datarate_seenpoint = 5.034674934438757
    # svd_precoder_datarate_unseenpoint = 4.9613323485265015
    svd_precoder_datarate_unseenpoint = 4.948339872265015
    plt.axhline(y=svd_precoder_datarate_seenpoint, color='g', linestyle='--', linewidth=2,
                label='SVD precoding datarate in static scene')
    plt.axhline(y=svd_precoder_datarate_unseenpoint, color='r', linestyle='--', linewidth=2,
                label='SVD precoding datarate in UE mobility test')

    # Set labels and title
    plt.xlabel('Data Augmentation Test Configuration (Dynamic) ', fontsize=11)
    plt.ylabel('Average Spectral Efficiency (bps/Hz)', fontsize=11)
    # plt.title('Spectral Efficiency with Different Data Augmentation Configurations', fontsize=16)

    # Set x and y axis limits
    plt.ylim(y_min, y_max)

    # Set the y-axis major locator to multiple of 0.02
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))

    # Restoring the complete axes and showing the plot
    sns.despine(offset={'left': 10, 'bottom': 2}, trim=False)  # offset表示坐标轴距离图像的距离，trim表示是否裁剪坐标轴
    plt.yticks(fontsize=10)
    plt.xticks([i + 1.5 * bar_width for i in index], df['Configuration'], fontsize=10, rotation=5)
    plt.grid(which='major', linestyle='-', linewidth='0.5')  # Thicker lines for major grid
    # Add legend
    plt.legend(loc='upper right', fontsize=8)

    # Adding lines at the top of each bar
    for i in index:
        x_coords = [i + j * bar_width for j in range(4)]
        y_coords = [df['No Augmentation'][i], df['VAE 100'][i], df['VAE 200'][i], df['VAE 400'][i]]
        plt.plot(x_coords, y_coords, marker='o', color=colors[3])

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    # Display plot
    plt.savefig('./fig/test_fig6_new.pdf')
    plt.show()


if __name__ == '__main__':
    # main()
    # draw()
    # draw2()
    # draw3()
    # draw4()
    # draw5()
    draw6()
