import random
import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio
from scipy import signal
from joblib import Parallel, delayed
import os

def baseline_correction(data):
    baseline_mean = np.mean(data, axis=0)  # 计算每个通道的基线平均值
    corrected_data = data - baseline_mean  # 对整个数据减去基线平均值
    return corrected_data

# # @滤波器组：FB4 Butter 6
# def fb_4_Butter6(x, num_channel, fs, l1, u1, l2, u2, l3, u3, l4, u4):
#     """
#     4个滤波器, 滤波范围: [l1, u1] (Hz) --- [l4, u4] (Hz)
#     """
#     # 参数计算
#     l1 = 2 * l1 / fs
#     u1 = 2 * u1 / fs
#
#     l2 = 2 * l2 / fs
#     u2 = 2 * u2 / fs
#
#     l3 = 2 * l3 / fs
#     u3 = 2 * u3 / fs
#
#     l4 = 2 * l4 / fs
#     u4 = 2 * u4 / fs
#
#     # 滤波1
#     channel_data_list1 = []
#     for i in range(num_channel):
#         b1, a1 = signal.butter(6, [l1, u1], 'bandpass')
#         filtedData1 = signal.filtfilt(b1, a1, x[:, i])
#         channel_data_list1.append(filtedData1)  # 变成chan_dim@time_dim
#     channel_data_list1 = np.array(channel_data_list1)
#
#     # 滤波2
#     channel_data_list2 = []
#     for i in range(num_channel):
#         b2, a2 = signal.butter(6, [l2, u2], 'bandpass')
#         filtedData2 = signal.filtfilt(b2, a2, x[:, i])
#         channel_data_list2.append(filtedData2)
#     channel_data_list2 = np.array(channel_data_list2)
#
#     # 滤波3
#     channel_data_list3 = []
#     for i in range(num_channel):
#         b3, a3 = signal.butter(6, [l3, u3], 'bandpass')
#         filtedData3 = signal.filtfilt(b3, a3, x[:, i])
#         channel_data_list3.append(filtedData3)
#     channel_data_list3 = np.array(channel_data_list3)
#
#     # 滤波4
#     channel_data_list4 = []
#     for i in range(num_channel):
#         b4, a4 = signal.butter(6, [l4, u4], 'bandpass')
#         filtedData4 = signal.filtfilt(b4, a4, x[:, i])
#         channel_data_list4.append(filtedData4)
#     channel_data_list4 = np.array(channel_data_list4)
#
#     return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4
#
# # @训练数据预处理方法
# def get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample, fs):
#     """
#     # path: 数据路径
#     # fs: 频率(下采样后的频率)
#     # down_sample: 下采样
#     # [wn11, wn21] ~ [wn14, wn24]: 4个滤波器，中括号[频率下限, 频率上限]
#     """
#     data = scio.loadmat(path)  # 读取原始数据
#
#     # 下采样与通道选择
#     x_data = data['EEG_SSVEP_train']['x'][0, 0][::down_sample]  # x_data变成了2维数数据，shape->Time_dim@Chan_dim
#     c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
#     train_data = x_data[:, c]
#     train_data = baseline_correction(train_data)
#     train_label = data['EEG_SSVEP_train']['y_dec'][0, 0][0]
#     train_start_time = data['EEG_SSVEP_train']['t'][0, 0][0]  # train_start_time变成1维数数据
#
#     channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4 \
#         = fb_4_Butter6(train_data, train_data.shape[1], fs, wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24)
#
#     return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, train_label, train_start_time


# ----------------- 向量化 & 使用 sosfiltfilt 的过滤器 -----------------
def fb_4_Butter6_fast(x, fs, bands):
    """
    x: shape (T, C)  时间在 axis=0，通道在 axis=1
    fs: 采样率（下采样后的频率）
    bands: [(l1,u1),(l2,u2),(l3,u3),(l4,u4)] 单位Hz
    返回 4 个数组，每个 shape (C, T) （保持与你原来代码一致：chan_dim@time_dim）
    """
    # 先把频率转换为数字频率（0..1）
    sos_list = []
    for (l, u) in bands:
        # 规范化频率并生成 sos
        low = 2.0 * l / fs
        high = 2.0 * u / fs
        sos = signal.butter(6, [low, high], btype='bandpass', output='sos')
        sos_list.append(sos)

    # 使用 sosfiltfilt 在 axis=0（时间轴）一次性过滤整个矩阵，返回 (T, C)
    filtered = []
    for sos in sos_list:
        # 注意：sosfiltfilt 需要 float64 数组以保证数值稳定性，但我们之后可以转换为 float32
        y = signal.sosfiltfilt(sos, x, axis=0, padtype='odd', padlen=None)  # 默认 padtype 较安全
        filtered.append(y.astype(np.float32))  # 转成 float32 减小内存

    # 转换为你原来期望的形状：channel_dim @ time_dim
    # 每个 filtered[i] 是 (T, C) -> 转置为 (C, T)
    return [arr.T for arr in filtered]  # 返回 list of 4 arrays: (C,T) each

# ----------------- 带缓存的 get_train_data_fast -----------------
def get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample, fs,
                        cache_dir='preprocessed', use_cache=True):
    """
    与原 get_train_data 功能等价，但做了向量化滤波与缓存。
    输入 path 是 mat 文件路径。
    返回：chan_data_list1..4 (每个 shape (C,T)), label, start_time
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    cache_file = os.path.join(cache_dir, f'{name}_fb4.npz')

    if use_cache and os.path.exists(cache_file):
        # 直接加载缓存
        data = np.load(cache_file)
        ch1 = data['ch1']
        ch2 = data['ch2']
        ch3 = data['ch3']
        ch4 = data['ch4']
        label = data['label'] if 'label' in data else None
        start_t = data['start_t'] if 'start_t' in data else None
        return ch1, ch2, ch3, ch4, label, start_t

    # 否则读取并计算
    mat = scio.loadmat(path)
    # 取训练数据路径下的结构字段（保持原来结构）
    x_data = mat['EEG_SSVEP_train']['x'][0,0][::down_sample]   # shape (T, all_channels)
    # 选择通道（保留 c 列）
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:, c]   # (T, C)
    train_data = baseline_correction(train_data)  # 如果你的函数接受 (T,C) 并返回同样形状

    train_label = mat['EEG_SSVEP_train']['y_dec'][0,0][0]
    train_start_time = mat['EEG_SSVEP_train']['t'][0,0][0]

    # 定义四个频带
    bands = [(wn11, wn21), (wn12, wn22), (wn13, wn23), (wn14, wn24)]
    ch1, ch2, ch3, ch4 = fb_4_Butter6_fast(train_data, fs, bands)  # each is (C,T)

    # 保存缓存以便下次直接加载
    if use_cache:
        np.savez_compressed(cache_file,
                            ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4,
                            label=np.array(train_label), start_t=np.array(train_start_time))
    return ch1, ch2, ch3, ch4, train_label, train_start_time

def get_test_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24,
                       path, down_sample, fs, cache_dir='preprocessed', use_cache=True):
    """
    快速版 get_test_data：向量化滤波 + 缓存
    返回：
        channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4,
        test_label, test_start_time
    其中每个 channel_data_list 是 shape (C, T) （与你原来代码一致：channel_dim @ time_dim）
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    cache_file = os.path.join(cache_dir, f'{name}_fb4_test.npz')

    if use_cache and os.path.exists(cache_file):
        data = np.load(cache_file)
        ch1 = data['ch1']
        ch2 = data['ch2']
        ch3 = data['ch3']
        ch4 = data['ch4']
        label = data['label'] if 'label' in data else None
        start_t = data['start_t'] if 'start_t' in data else None
        return ch1, ch2, ch3, ch4, label, start_t

    # load .mat
    mat = scio.loadmat(path)

    x_data = mat['EEG_SSVEP_test']['x'][0, 0][::down_sample]   # (T, all_ch)
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    test_data = x_data[:, c]   # (T, C)
    test_data = baseline_correction(test_data)  # 保持 (T,C) 输入/输出

    test_label = mat['EEG_SSVEP_test']['y_dec'][0, 0][0]

    test_start_time = mat['EEG_SSVEP_test']['t'][0, 0][0]

    # 定义频带并调用向量化滤波（fb_4_Butter6_fast）
    bands = [(wn11, wn21), (wn12, wn22), (wn13, wn23), (wn14, wn24)]
    ch1, ch2, ch3, ch4 = fb_4_Butter6_fast(test_data, fs, bands)  # 返回 (C, T) each

    # 保存缓存以便下次快速加载
    if use_cache:
        np.savez_compressed(cache_file,
                            ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4,
                            label=np.array(test_label), start_t=np.array(test_start_time))

    return ch1, ch2, ch3, ch4, test_label, test_start_time
# # @测试数据预处理方法
# def get_test_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample, fs):
#     """
#     # path: 数据路径
#     # fs: 频率(下采样后的频率)
#     # down_sample: 下采样
#     # [wn11, wn21] ~ [wn14, wn24]: 4个滤波器，中括号[频率下限, 频率上限]
#     """
#     data = scio.loadmat(path)  # 读取原始数据
#
#     # 下采样与通道选择
#     x_data = data['EEG_SSVEP_test']['x'][0][0][::down_sample]
#     c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
#     test_data = x_data[:, c]
#     test_data = baseline_correction(test_data)
#     test_label = data['EEG_SSVEP_test']['y_dec'][0][0][0]
#     test_start_time = data['EEG_SSVEP_test']['t'][0][0][0]
#
#     channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4 \
#         = fb_4_Butter6(test_data, test_data.shape[1], fs, wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24)
#
#     return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, test_label, test_start_time


class train_BCIDataset(Dataset):
    def __init__(self, num_data, train_data, win_train, y_label, start_time, down_sample, channel):
        super(train_BCIDataset, self).__init__()

        x_train, y_train = list(range(num_data)), list(range(num_data))  # 创建数据集合容器

        # 从90组实验中，随机重复提取num_data组数据
        for i in range(num_data):
            k = random.randint(0, (y_label.shape[0] - 1))  # 随机选择一组实验, 加上[0]的目的是让其从list变成个number
            y_data = y_label[k] - 1  # 获取标签

            time_start = random.randint(35, int(1000 + 35 - win_train))  # 随机时间,下采样后的频率，250x4=1000

            t1 = int(start_time[k] / down_sample) + time_start
            t2 = int(start_time[k] / down_sample) + time_start + win_train
            train_data = np.array(train_data)
            x_1 = train_data[:, :, t1:t2]
            x_2 = np.reshape(x_1, (4, channel, win_train))  # eg. 4*9*50
            x_train[i] = x_2.astype(np.float32)  # pytorch 参数是float32，故输入数据的x需要求float32
            y_train[i] = y_data

        self.data = x_train
        self.label = y_train
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]


class val_BCIDataset(Dataset):
    def __init__(self, num_data, test_data, win_train, y_label, start_time, down_sample, channel):
        super(val_BCIDataset, self).__init__()

        x_test, y_test = list(range(num_data)), list(range(num_data))

        # 从100组实验中，随机重复提取num_data组数据
        for i in range(num_data):
            k = random.randint(0, (y_label.shape[0] - 1))  # 随机选择一组实验, 加上[0]的目的是让其从list变成个number
            y_data = y_label[k] - 1  # 获取标签

            time_start = random.randint(35, int(1000 + 35 - win_train))  # 随机时间

            t1 = int(start_time[k] / down_sample) + time_start
            t2 = int(start_time[k] / down_sample) + time_start + win_train
            test_data = np.array(test_data)
            x_1 = test_data[:, :, t1:t2]
            x_2 = np.reshape(x_1, (4, channel, win_train))  # eg. 4*9*50

            x_test[i] = x_2.astype(np.float32)  # pytorch 参数是float32，故输入数据的x需要求float32
            y_test[i] = y_data

        self.data = x_test
        self.label = y_test
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]

