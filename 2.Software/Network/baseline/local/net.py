import torch
import torch.nn as nn
import config as config
from einops import rearrange

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
        self.conv1D_block_2 = conv1D_block_(out_channel, out_channel, (1, 25), 1, drop_out1)
        self.conv1D_block_3 = conv1D_block_(out_channel, out_channel, (1, 11), 1, drop_out1)

    def forward(self, x):
        x1 = self.conv1D_block_1(x)
        x2 = self.conv1D_block_1(x)
        x3 = self.conv1D_block_1(x)

        x2_2 = self.conv1D_block_2(x2)
        x3_2 = x3 + x2_2
        x3_3 = self.conv1D_block_3(x3_2)

        y = x1 + x2_2 + x3_3
        return y

class MultiScaleCNN(nn.Module):
    def __init__(self, num_fb, channel, drop_rate):
        super(MultiScaleCNN, self).__init__()
        self.cov2d1 = nn.Conv2d(num_fb, num_fb, (channel, 1), 1)  # 卷后形状为8@1*250
        self.bn1_1 = nn.BatchNorm2d(num_fb, momentum=0.99, eps=0.001)
        self.elu_1 = nn.ELU()

        self.multi_scale_1D = multi_scale_1D(num_fb, 2 * num_fb, drop_rate)

        self.cov2d2 = nn.Conv2d(2 * num_fb, channel, (1, 1), 1)
        self.bn1_2 = nn.BatchNorm2d(channel, momentum=0.99, eps=0.001)
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
        # print(x.shape)
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
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

        # self.dropout = nn.Dropout(config.drop_rate)
        self.linear = nn.Linear(config.channel * config.win_data, config.win_data)
        self.ln = nn.LayerNorm(config.win_data)
        self.gelu = nn.GELU()
        # self.dropout2 = nn.Dropout(config.drop_rate)
        self.linear2 = nn.Linear(config.win_data, config.n_feature)

    def forward(self, x):
        x = self.cnn_blk(x)
        x = torch.squeeze(x)
        x = self.transformer(x)
        x = torch.flatten(x, 1, 2)

        # x = self.dropout(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        # x = self.dropout2(x)
        out = self.linear2(x)
        return out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(config.drop_rate)
        self.linear = nn.Linear(config.n_feature, 6 * config.n_class)
        self.ln = nn.LayerNorm(6 * config.n_class)
        self.gelu = nn.GELU()
        self.dropout2 = nn.Dropout(config.drop_rate)
        self.linear2 = nn.Linear(6 * config.n_class, config.n_class)

    def forward(self, x):
        # x = torch.flatten(x, 1, 2)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        out = self.linear2(x)

        return out


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