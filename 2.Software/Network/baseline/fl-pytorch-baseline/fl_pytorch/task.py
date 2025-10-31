"""FL-PyTorch: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fl_pytorch.config as config
from fl_pytorch.dataloader import *
from einops import rearrange
import torch.nn.functional as F
from collections import defaultdict

class conv1D_block_(nn.Module):
    def __init__(self, in_channel, out_channel, k_size, stride, drop_rate):
        super(conv1D_block_, self).__init__()
        self.dropout_1 = nn.Dropout(drop_rate)
        self.cnn_cov1d = nn.Conv2d(in_channel, out_channel, k_size, stride, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=0.99, eps=0.001)
        self.elu = nn.ELU(alpha=1)

    def forward(self, x):
        x = self.dropout_1(x)
        x = self.cnn_cov1d(x)
        x = self.bn1(x)
        y = self.elu(x)
        return y

class multi_scale_1D(nn.Module):
    def __init__(self, in_channel, out_channel, drop_out1):
        super(multi_scale_1D, self).__init__()
        self.conv1D_block_1 = conv1D_block_(in_channel, out_channel, 1, 1, drop_out1)
        self.conv1D_block_2 = conv1D_block_(out_channel, out_channel, (1, 32), 1, drop_out1)
        self.conv1D_block_3 = conv1D_block_(out_channel, out_channel, (1, 16), 1, drop_out1)
        self.conv1D_block_4 = conv1D_block_(out_channel, out_channel, (1, 11), 1, drop_out1)

    def forward(self, x):
        x1 = self.conv1D_block_1(x)
        x2 = self.conv1D_block_1(x)
        x3 = self.conv1D_block_1(x)
        x4 = self.conv1D_block_1(x)

        x2_2 = self.conv1D_block_2(x2)
        x3_2 = x3 + x2_2
        x3_3 = self.conv1D_block_3(x3_2)
        x4_3 = x4 + x3_3
        x4_4 = self.conv1D_block_4(x4_3)

        y = x1 + x2_2 + x3_3 + x4_4
        return y

class MultiScaleCNN(nn.Module):
    def __init__(self, num_fb, channel, drop_rate):
        super(MultiScaleCNN, self).__init__()
        self.cov2d1 = nn.Conv2d(num_fb, 8, (channel, 1), 1)  # 卷后形状为8@1*250
        self.bn1_1 = nn.BatchNorm2d(8, momentum=0.99, eps=0.001)
        self.elu_1 = nn.ELU()

        self.multi_scale_1D = multi_scale_1D(8, 32, drop_rate)

        self.cov2d2 = nn.Conv2d(32, 18, (1, 1), 1)
        self.bn1_2 = nn.BatchNorm2d(18, momentum=0.99, eps=0.001)
        self.elu_2 = nn.ELU()

    def forward(self, x):
        x = self.cov2d1(x)
        x = self.bn1_1(x)
        x = self.elu_1(x)

        x = self.multi_scale_1D(x)

        x = self.cov2d2(x)
        x = self.bn1_2(x)
        y = self.elu_2(x)

        return y

class Attention(nn.Module):
    # dim: 输入张量的最后一维大小
    # heads：多头注意力机制中的头数，默认为 8。
    # dim_head：每个头的维度，默认为 64。
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，用于将计算得到的 q 和 k 的点积进行缩放，避免数值过大，1/sqrt(dim_head)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        te = self.to_qkv(x)
        qkv = te.chunk(3, dim=-1)  # 依次切分成三份
        # 张量形状转换
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # q点乘k的转置
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        # 加权注意力
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 定义 Layer Normalization 对输入数据的最后一维进行归一化操作
        self.fn = fn  # 接收一个函数（通常是一个子模块，例如 Attention 或 FeedForward）

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    # dim: 输入张量的最后一维大小
    # depth: Transformer的深度
    # heads：多头注意力机制中的头数
    # dim_head：每个头的维度
    # mlp_dim： 前馈网络隐藏层深度
    # dropout：dropout 比例，默认为 0.1。
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                               PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# 特征提取器：包括 CNN 和 Transformer 部分
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.cnn_blk = MultiScaleCNN(config.num_fb, config.channel, config.drop_rate)
        self.transformer = Transformer(config.win_data, 1, 8, 128, 256, config.drop_rate)

        self.dropout = nn.Dropout(config.drop_rate)
        self.linear = nn.Linear(18 * config.win_data, config.win_data)
        self.ln = nn.LayerNorm(config.win_data)
        self.gelu = nn.GELU()
        self.dropout2 = nn.Dropout(config.drop_rate)
        self.linear2 = nn.Linear(config.win_data, config.n_feature)

    def forward(self, x):
        x = self.cnn_blk(x)
        x = x.squeeze(2)  # 指定去掉第三个维度
        x = self.transformer(x)
        x = torch.flatten(x, 1, 2)

        x = self.dropout(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        out = self.linear2(x)
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(config.drop_rate_g)
        self.linear = nn.Linear(config.n_feature, 6 * config.n_class)
        self.ln = nn.LayerNorm(6 * config.n_class)
        self.elu = nn.ELU()
        self.dropout2 = nn.Dropout(config.drop_rate_g)
        self.linear2 = nn.Linear(6 * config.n_class, config.n_class)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.elu(x)
        x = self.dropout2(x)
        out = self.linear2(x)

        return out


class FeatureGenerator(nn.Module):
    def __init__(self):
        super(FeatureGenerator, self).__init__()
        self.linear = nn.Linear(config.n_class, 256)
        self.ln = nn.LayerNorm(256)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(256, config.n_feature)

    def forward(self, labels):
        batch_size = labels.size(0)
        # 创建一个形状为 batch_size × n_classes 的全零张量
        y_input = torch.FloatTensor(batch_size, config.n_class).to(labels.device)
        y_input.zero_()
        # 将 labels 对应的索引处设置为1
        y_input.scatter_(1, labels.view(-1, 1), 1)

        x = y_input

        x = self.linear(x)
        x = self.ln(x)
        x = self.elu(x)
        out = self.linear2(x)

        return out



# 将特征提取器和分类器组合
class TwinBranchNets(nn.Module):
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        super(TwinBranchNets, self).__init__()
        self.feature_extractor = feature_extractor  # 特征提取部分
        self.classifier = classifier  # 分类部分

    def forward(self, x):
        # 先通过特征提取器
        feature = self.feature_extractor(x)
        # 再通过分类器
        x = self.classifier(feature)
        return x

class VanillaKDLoss(nn.Module):
    def __init__(self, temperature):
        super(VanillaKDLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        loss = nn.functional.kl_div(nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
                                    nn.functional.softmax(teacher_logits / self.temperature, dim=-1),
                                    reduction='batchmean') * self.temperature * self.temperature
        return loss


# 返回训练集和测试集
def load_data(partition_id: int):
    """Load partitioned EEG data for federated learning clients."""
    # 根据 partition_id 加载对应的文件
    if partition_id < 9:
        path_sub = f'/home/rao/Data/sess01/sess01_subj0{partition_id + 1}_EEG_SSVEP.mat'  # +1 是为了从1开始
    else:
        path_sub = f'/home/rao/Data/sess01/sess01_subj{partition_id + 1}_EEG_SSVEP.mat'

    # if partition_id < 9:
    #     path_sub = f'C:\\Data\\sess01\\sess01_subj0{partition_id + 1}_EEG_SSVEP.mat'  # +1 是为了从1开始
    # else:
    #     path_sub = f'C:\\Data\\sess01\\sess01_subj{partition_id + 1}_EEG_SSVEP.mat'

    # 获取滤波后的训练数据、标签和起始时间
    data_tra_tmp1, data_tra_tmp2, data_tra_tmp3, data_tra_tmp4, label_tra, start_time_tra = get_train_data(
        config.f_down1, config.f_up1, config.f_down2, config.f_up2, config.f_down3, config.f_up3,
        config.f_down4, config.f_up4, path_sub, config.down_sample, config.fs)
    data_tra = [data_tra_tmp1, data_tra_tmp2, data_tra_tmp3, data_tra_tmp4]
    data_tra = np.array(data_tra)
    # 获取滤波后的测试数据、标签和起始时间
    # data_tes_tmp1, data_tes_tmp2, data_tes_tmp3, data_tes_tmp4, label_tes, start_time_tes = get_test_data(
    #     config.f_down1, config.f_up1, config.f_down2, config.f_up2, config.f_down3, config.f_up3,
    #     config.f_down4, config.f_up4, path_sub, config.down_sample, config.fs)
    # data_tes = [data_tes_tmp1, data_tes_tmp2, data_tes_tmp3, data_tes_tmp4]
    # data_tes = np.array(data_tes)
    # 创建数据集及加载器
    dataset_tra = train_BCIDataset(config.num_data, data_tra, config.win_data, label_tra, start_time_tra, config.down_sample, config.channel)
    loader_tra = DataLoader(dataset_tra, shuffle=True, batch_size=config.bth_size, num_workers=1, pin_memory=True, drop_last=True)
    # dataset_tes = test_BCIDataset(config.num_data, data_tes, config.win_data, label_tes, start_time_tes, config.down_sample, config.channel)
    # loader_tes = DataLoader(dataset_tes, shuffle=True, batch_size=config.bth_size, num_workers=1, pin_memory=True, drop_last=True)
    # print('客户端%d获取数据' % partition_id)
    return loader_tra, loader_tra


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.Adam(net.parameters(), config.lr, weight_decay=0.001)  # 对参数进行正则化weight_decay, 防止过拟合
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0  # 每个 epoch 的累积损失
        for iter_idx_tra, (data_tra_bth, tgt_tra_bth) in enumerate(trainloader):
            data_tra_bth, tgt_tra_bth = data_tra_bth.to(device), tgt_tra_bth.to(device)
            optimizer.zero_grad()  # 优化器梯度清零
            output_tra_bth = net(data_tra_bth)
            loss_tra_bth = criterion(output_tra_bth, tgt_tra_bth.long())
            loss_tra_bth.backward()
            optimizer.step()
            epoch_loss += loss_tra_bth.item()
        running_loss += epoch_loss
    # 计算所有 epochs 的平均训练损失
    avg_trainloss = running_loss / len(trainloader) / epochs
    return avg_trainloss

def train_Prox(net, trainloader, epochs, device, mu = 0.01):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.Adam(net.parameters(), config.lr, weight_decay=0.001)  # 对参数进行正则化weight_decay, 防止过拟合
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.train()
    running_loss = 0.0
    global_params = [param.clone().detach() for param in net.parameters()]
    for epoch in range(epochs):
        epoch_loss = 0.0  # 每个 epoch 的累积损失
        for iter_idx_tra, (data_tra_bth, tgt_tra_bth) in enumerate(trainloader):
            data_tra_bth, tgt_tra_bth = data_tra_bth.to(device), tgt_tra_bth.to(device)
            optimizer.zero_grad()  # 优化器梯度清零
            output_tra_bth = net(data_tra_bth)
            loss_tra_bth = criterion(output_tra_bth, tgt_tra_bth.long())
            # 加入 FedProx 正则项
            prox_term = 0.0
            for param, global_param in zip(net.parameters(), global_params):
                prox_term += ((param - global_param) ** 2).sum()
            loss_tra_bth += (mu / 2) * prox_term
            loss_tra_bth.backward()
            optimizer.step()
            epoch_loss += loss_tra_bth.item()
        running_loss += epoch_loss
    # 计算所有 epochs 的平均训练损失
    avg_trainloss = running_loss / len(trainloader) / epochs
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    net.eval()
    for iter_idx_tes, bth_tes in enumerate(testloader):
        data_tes_bth, tgt_tes_bth = bth_tes[0], bth_tes[1]
        data_tes_bth, tgt_tes_bth = data_tes_bth.to(device), tgt_tes_bth.to(device)
        with torch.no_grad():  # 取消梯度计算环节
            output_tes_bth = net(data_tes_bth)
            loss += criterion(output_tes_bth, tgt_tes_bth.long()).item()

            # 获取每个样本的预测概率
            probabilities = nn.functional.softmax(output_tes_bth, dim=-1).cpu().numpy()
            all_probabilities.append(probabilities)  # 按批次收集概率

            # 获取每个样本的预测标签（概率最大值对应的类别）
            predictions = torch.max(output_tes_bth.data, 1)[1]

            correct += (predictions == tgt_tes_bth).sum().item()

            # 收集真实标签、预测标签和预测概率
            all_targets.extend(tgt_tes_bth.cpu().numpy().tolist())  # 转为列表
            all_predictions.extend(predictions.cpu().numpy().tolist())

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    # 将所有批次的概率拼接成一个矩阵
    all_probabilities = np.vstack(all_probabilities)
    # print('客户端%d测试准确率%.3f 损失值%.3f' % (partition_id, accuracy, loss))
    # 返回损失、准确率、真实标签、预测标签和预测概率
    return loss, accuracy, all_targets, all_predictions, all_probabilities

def get_weights(net):
    # state_dict()获取模型参数  它返回一个 字典（OrderedDict）
    # net.state_dict().items() 返回模型 net 的参数字典中的每一对键值对（键是参数的名称，值是参数的张量）。
    # for _, val 表示我们遍历每个键值对，_ 用来占位，因为在这个上下文中我们只关心 val（参数的张量），不需要用到键（参数的名称），所以使用 _ 来忽略键。
    # 列表推导式语法 [表达式 for 元素 in 可迭代对象]
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    # 将 `net.state_dict().keys()` 和 `parameters` 进行配对，将参数对应到网络的每一层
    # zip() 函数将模型的权重名称（state_dict().keys()）和 parameters 进行一一配对，形成一个可迭代对象
    params_dict = zip(net.state_dict().keys(), parameters)
    # 通过列表推导式，我们将每个模型的权重名称 k 与相应的权重值 v 转换成一个新的 torch.tensor 对象，形成新的字典。
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # 将新创建的 state_dict（包含了服务器同步过来的权重）加载到本地的模型 net 中
    net.load_state_dict(state_dict, strict=True)

