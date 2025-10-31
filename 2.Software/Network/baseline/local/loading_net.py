import torch
import numpy as np
from net import tCNN  # 导入你的模型类
import scipy.io as scio
from scipy import signal
from dataloader import train_BCIDataset, val_BCIDataset, test_BCIDataset
from torch.utils.data import DataLoader
import serial  # 导入模块
# @获取滤波后的训练数据，标签和起始时间
def get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path, down_sample):
    data = scio.loadmat(path)  # 读取原始数据
    # 下采样与通道选择
    x_data = data['EEG_SSVEP_train']['x'][0][0][::down_sample]
    c = [24, 28, 29, 30, 41, 42, 43, 60, 61]    # 这九个数是对应传感器电极片的位置 即九个通道
    train_data = x_data[:, c]
    train_label = data['EEG_SSVEP_train']['y_dec'][0][0][0]
    train_start_time = data['EEG_SSVEP_train']['t'][0][0][0]
    # @ 滤波1
    channel_data_list1 = []
    for i in range(train_data.shape[1]):    # 循环遍历 train_data 的每一列
        b, a = signal.butter(6, [wn11, wn21], 'bandpass')   # 带通滤波器
        filtedData = signal.filtfilt(b, a, train_data[:, i])  # :表示所有行  train_data[:, i] 就是选取了所有样本在第 i 个特征（列）上的数据。
        channel_data_list1.append(filtedData)      # 将每个通道的滤波后数据添加到列表中
    channel_data_list1 = np.array(channel_data_list1)  # 将列表转换为 NumPy 数组  每一行表示一个通道的滤波后数据
    # @ 滤波2
    channel_data_list2 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn12, wn22], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list2.append(filtedData)
    channel_data_list2 = np.array(channel_data_list2)
    # @ 滤波3
    channel_data_list3 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn13, wn23], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list3.append(filtedData)
    channel_data_list3 = np.array(channel_data_list3)
    # @ 滤波4
    channel_data_list4 = []
    for i in range(train_data.shape[1]):
        b, a = signal.butter(6, [wn14, wn24], 'bandpass')
        filtedData = signal.filtfilt(b, a, train_data[:, i])
        channel_data_list4.append(filtedData)
    channel_data_list4 = np.array(channel_data_list4)

    return channel_data_list1, channel_data_list2, channel_data_list3, channel_data_list4, train_label, train_start_time

def Port_send(order):
    # 串口初始化
    portx = 'COM7'  # 端口号
    bps = 115200  # 波特率
    timex = 5  # 超时设置
    ser = serial.Serial(portx, bps, timeout=timex)  # 打开串口,并获得实例对象

    if order == 0:
        x = bytearray(b'\xaa\xbb\x00\xa0\x00\x00\xcc\xdd')
        ser.write(x)
        ser.close()
        print('Open!')
    elif order == 1:
        x = bytearray(b'\xaa\xbb\x00\x00\x00\x00\xcc\xdd')
        ser.write(x)
        ser.close()
        print('Close!')
    elif order == 2:
        x = bytearray(b'\xaa\xbb\x00\x32\x00\x00\xcc\xdd')
        ser.write(x)
        ser.close()
        print('Half Open!')
    elif order == 3:
        x = bytearray(b'\xaa\xbb\x00\x64\x00\x00\xcc\xdd')
        ser.write(x)
        ser.close()
        print('Half Close!')
    else:
        print('No Send!')


if __name__ == '__main__':
    win_tim = 0.2  # 时间窗口0.2s

    num_data = 1  # 数据集的大小

    down_sample = 4  # 下采样设置
    fs = 1000 / down_sample  # fs为float类型   每秒的样本数
    channel = 9  # 选取的通道数

    f_down1 = 3  # 第一个滤波器 下限和上限频率
    f_up1 = 14
    wn11 = 2 * f_down1 / fs  # 第一个滤波器的归一化频率范围 通过将频率除以采样率来计算
    wn21 = 2 * f_up1 / fs

    f_down2 = 9  # 第二个滤波器
    f_up2 = 26
    wn12 = 2 * f_down2 / fs
    wn22 = 2 * f_up2 / fs

    f_down3 = 14  # 第三个滤波器
    f_up3 = 38
    wn13 = 2 * f_down3 / fs
    wn23 = 2 * f_up3 / fs

    f_down4 = 19  # 第四个滤波器
    f_up4 = 50
    wn14 = 2 * f_down4 / fs
    wn24 = 2 * f_up4 / fs
    win_data = int(fs * win_tim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = tCNN(win_data)  # 假设你的模型类是 tCNN，需要根据实际情况修改
    model.to(device)
    print('Loading weights into state dict ...')

    # 当保存的模型是GPU训练的，而现在我们要用CPU验证时需要添加这样一句话
    # torch.load() 函数加载保存的模型参数
    # load_state_dict() 函数把加载的参数赋值给模型
    model.load_state_dict(torch.load('your_model.pth', map_location=device))
    model.eval()  # 设置模型为评估模式

    win_data = int(fs * win_tim)  # 时间窗口对应数据帧数  50

    path_sub = f'D:\\sess01\\sess01_subj01_EEG_SSVEP.mat'

    mid_data_tra_1, mid_data_tra_2, mid_data_tra_3, mid_data_tra_4, label_tra, start_time_tra \
        = get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, wn14, wn24, path_sub, down_sample)
    data_tra = [mid_data_tra_1, mid_data_tra_2, mid_data_tra_3, mid_data_tra_4]  # 数据聚合, 形状变为4*9*50
    data_tra = np.array(data_tra)

    list_tra = [2]
    list_tra = [0]
    list_tra = [7]
    list_tra = [3]


    dataset_tra = train_BCIDataset(num_data, data_tra, win_data, label_tra, start_time_tra,
                                   down_sample, list_tra, channel)
    gen_tra = DataLoader(dataset_tra, shuffle=True, batch_size=num_data, num_workers=1,
                         pin_memory=True, drop_last=True)

    for iter_tra, batch in enumerate(gen_tra):
        if iter_tra >= num_data:
            break
        data_test, targets = batch[0], batch[1]
        data_test, targets = data_test.to(device), targets.to(device)
        # 进行预测
        with torch.no_grad():  # 不需要计算梯度
            output = model(data_test)  # 使用转换后的张量进行预测

        predictions = torch.argmax(output, dim=1)  # 获取预测结果

        # 打印预测结果
        print("Predictions:", predictions.item())
        print("Targets:",  targets.item())
        print("Output:", output)
        # Port_send(targets.item())

