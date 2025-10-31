import random
import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio
from scipy import signal

def baseline_correction(data):
    baseline_mean = np.mean(data, axis=0)  # 计算每个通道的基线平均值
    corrected_data = data - baseline_mean  # 对整个数据减去基线平均值
    return corrected_data

def normalize_channels(data):
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True)
    data_stand = (data - mean) / std
    return data_stand

# @滤波器组：FB4 Butter 6
def fb_4_Butter6(x, num_channel, fs, l1, u1, l2, u2, l3, u3, l4, u4):
    """
    4个滤波器, 滤波范围: [l1, u1] (Hz) --- [l4, u4] (Hz)
    """
    # 参数计算
    l1 = 2 * l1 / fs
    u1 = 2 * u1 / fs

    l2 = 2 * l2 / fs
    u2 = 2 * u2 / fs

    l3 = 2 * l3 / fs
    u3 = 2 * u3 / fs

    l4 = 2 * l4 / fs
    u4 = 2 * u4 / fs

    # 滤波1
    channel_data_list1 = []
    for i in range(num_channel):
        b1, a1 = signal.butter(6, [l1, u1], 'bandpass')
        filtedData1 = signal.filtfilt(b1, a1, x[:, i])
        channel_data_list1.append(filtedData1)  # 变成chan_dim@time_dim
    channel_data_list1 = np.array(channel_data_list1)

    # 滤波2
    channel_data_list2 = []
    for i in range(num_channel):
        b2, a2 = signal.butter(6, [l2, u2], 'bandpass')
        filtedData2 = signal.filtfilt(b2, a2, x[:, i])
        channel_data_list2.append(filtedData2)
    channel_data_list2 = np.array(channel_data_list2)

    # 滤波3
    channel_data_list3 = []
    for i in range(num_channel):
        b3, a3 = signal.butter(6, [l3, u3], 'bandpass')
        filtedData3 = signal.filtfilt(b3, a3, x[:, i])
        channel_data_list3.append(filtedData3)
    channel_data_list3 = np.array(channel_data_list3)

    # 滤波4
    channel_data_list4 = []
    for i in range(num_channel):
        b4, a4 = signal.butter(6, [l4, u4], 'bandpass')
        filtedData4 = signal.filtfilt(b4, a4, x[:, i])
        channel_data_list4.append(filtedData4)
    channel_data_list4 = np.array(channel_data_list4)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4


# @训练数据预处理方法
def get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample, fs):
    data = scio.loadmat(path)  # 读取原始数据
    # 下采样与通道选择
    x_data = data['EEG_SSVEP_train']['x'][0, 0]  # x_data变成了2维数数据，shape->Time_dim@Chan_dim
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    train_data = x_data[:, c]
    train_data = baseline_correction(train_data)[::down_sample]
    # train_data = standardize_data(train_data)
    train_label = data['EEG_SSVEP_train']['y_dec'][0, 0][0]
    train_start_time = data['EEG_SSVEP_train']['t'][0, 0][0]  # train_start_time变成1维数数据

    channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4 \
        = fb_4_Butter6(train_data, train_data.shape[1], fs, wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, train_label, train_start_time


# @测试数据预处理方法
def get_test_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample, fs):
    data = scio.loadmat(path)  # 读取原始数据
    # 下采样与通道选择
    x_data = data['EEG_SSVEP_test']['x'][0][0]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]
    test_data = x_data[:, c]
    test_data = baseline_correction(test_data)[::down_sample]
    # test_data = standardize_data(test_data)
    test_label = data['EEG_SSVEP_test']['y_dec'][0][0][0]
    test_start_time = data['EEG_SSVEP_test']['t'][0][0][0]

    channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4 \
        = fb_4_Butter6(test_data, test_data.shape[1], fs, wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, test_label, test_start_time

class train_BCIDataset(Dataset):
    def __init__(self, num_data, train_data, win_train, y_label, start_time, down_sample, channel):
        super(train_BCIDataset, self).__init__()

        x_train, y_train = list(range(num_data)), list(range(num_data))  # 创建数据集合容器

        # 从90组实验中，随机重复提取num_data组数据
        for i in range(num_data):
            k = random.randint(0, (y_label.shape[0] - 1)) # 随机选择一组实验, 加上[0]的目的是让其从list变成个number
            y_data = y_label[k] - 1  # 获取标签

            time_start = random.randint(35, int(1000 + 35 - win_train))  # 随机时间,下采样后的频率，250x4=1000

            t1 = int(start_time[k] / down_sample) + time_start
            t2 = int(start_time[k] / down_sample) + time_start + win_train
            x_1 = train_data[:, :, t1:t2]
            x_2 = np.reshape(x_1, (4, channel, win_train))  # eg. 4*9*50
            x_2 = normalize_channels(x_2)
            x_train[i] = x_2.astype(np.float32)  # pytorch 参数是float32，故输入数据的x需要求float32
            y_train[i] = y_data

        self.data = x_train
        self.label = y_train
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]


class test_BCIDataset(Dataset):
    def __init__(self, num_data, test_data, win_train, y_label, start_time, down_sample, channel):
        super(test_BCIDataset, self).__init__()
        x_test, y_test = list(range(num_data)), list(range(num_data))

        # 从100组实验中，随机重复提取num_data组数据
        for i in range(num_data):
            k = random.randint(0, (y_label.shape[0] - 1))  # 随机选择一组实验, 加上[0]的目的是让其从list变成个number
            y_data = y_label[k] - 1  # 获取标签

            time_start = random.randint(35, int(1000 + 35 - win_train))  # 随机时间

            t1 = int(start_time[k] / down_sample) + time_start
            t2 = int(start_time[k] / down_sample) + time_start + win_train
            x_1 = test_data[:, :, t1:t2]
            x_2 = np.reshape(x_1, (4, channel, win_train))  # eg. 4*9*50
            x_2 = normalize_channels(x_2)
            x_test[i] = x_2.astype(np.float32)  # pytorch 参数是float32，故输入数据的x需要求float32
            y_test[i] = y_data

        self.data = x_test
        self.label = y_test
        self.num_total = num_data

    def __len__(self):
        return self.num_total

    def __getitem__(self, idk):
        return self.data[idk], self.label[idk]

