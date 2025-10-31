"""
This is a standard script for the FBtCNNDing2021
"""
# @网络参数
num_fb = 4
# C = 9
# F = 250
drop_rate = 0.4
n_class = 4
n_feature = 20

# @数据预处理参数
tw = 0.8
down_sample = 4  # 下采样设置
fs = 1000 / down_sample  # fs为float类型
win_data = int(fs * tw)

f_down1 = 3  # 第一个滤波器
f_up1 = 14

f_down2 = 9  # 第二个滤波器
f_up2 = 26

f_down3 = 14  # 第三个滤波器
f_up3 = 38

f_down4 = 19  # 第四个滤波器
f_up4 = 50


# @训练参数
num_sub = 54
num_rounds = 50
num_epo = 5
num_data = 4096  # 512 1024 5120 10240
bth_size = 256
channel = 9  # 选取的通道数
lr = 4e-4



