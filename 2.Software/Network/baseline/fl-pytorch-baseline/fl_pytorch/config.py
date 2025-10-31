# @网络参数
num_fb = 4  # 滤波器层数
drop_rate = 0.4  # 丢弃率
drop_rate_g = 0.2
n_class = 4  # 分类数
channel = 9  # 选取的通道数
n_feature = 20

# @数据预处理参数
tw = 1  # 时间窗
down_sample = 4  # 下采样设置
fs = 1000 / down_sample  # 帧数
win_data = int(fs * tw)  # 时间窗口对应数据长度

f_down1 = 3  # 第一个滤波器
f_up1 = 14

f_down2 = 9  # 第二个滤波器
f_up2 = 26

f_down3 = 14  # 第三个滤波器
f_up3 = 38

f_down4 = 19  # 第四个滤波器
f_up4 = 50

# @训练参数
num_data = 4096
bth_size = 256
lr = 4e-4