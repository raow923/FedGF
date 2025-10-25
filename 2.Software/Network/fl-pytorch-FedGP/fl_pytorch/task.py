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
        self.cov2d1 = nn.Conv2d(num_fb, 8, (channel, 1), 1)
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        te = self.to_qkv(x)
        qkv = te.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
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
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
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
        x = x.squeeze(2)
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
        y_input = torch.FloatTensor(batch_size, config.n_class).to(labels.device)
        y_input.zero_()
        y_input.scatter_(1, labels.view(-1, 1), 1)

        x = y_input

        x = self.linear(x)
        x = self.ln(x)
        x = self.elu(x)
        out = self.linear2(x)

        return out


class TwinBranchNets(nn.Module):
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        super(TwinBranchNets, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        feature = self.feature_extractor(x)
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
    if partition_id < 9:
        path_sub = f'/home/rao/Data/sess01/sess01_subj0{partition_id + 1}_EEG_SSVEP.mat'
    else:
        path_sub = f'/home/rao/Data/sess01/sess01_subj{partition_id + 1}_EEG_SSVEP.mat'

    data_tra_tmp1, data_tra_tmp2, data_tra_tmp3, data_tra_tmp4, label_tra, start_time_tra = get_train_data(
        config.f_down1, config.f_up1, config.f_down2, config.f_up2, config.f_down3, config.f_up3,
        config.f_down4, config.f_up4, path_sub, config.down_sample, config.fs)
    data_tra = [data_tra_tmp1, data_tra_tmp2, data_tra_tmp3, data_tra_tmp4]
    data_tra = np.array(data_tra)

    dataset_tra = train_BCIDataset(config.num_data, data_tra, config.win_data, label_tra, start_time_tra,
                                   config.down_sample, config.channel)
    loader_tra = DataLoader(dataset_tra, shuffle=True, batch_size=config.bth_size, num_workers=1, pin_memory=True,
                            drop_last=True)

    return loader_tra, loader_tra


def negative_interpolation(u1, u2, lambda_val=0.1):
    return (u2 - u1) * lambda_val + u1

def angle_loss(f, u_plus, tau=1):
    total_angle_loss = 0
    n = f.size(0)
    for i in range(n):
        for j in range(i + 1, n):
            cos_f = F.cosine_similarity(f[i], f[j], dim=0) / tau
            cos_u = F.cosine_similarity(u_plus[i], u_plus[j], dim=0) / tau

            total_angle_loss += torch.abs(cos_f - cos_u)

    return total_angle_loss / (n * (n - 1) / 2)

def edge_loss(f, u_plus):
    total_edge_loss = 0
    n = f.size(0)
    for i in range(n):
        for j in range(i + 1, n):
            f_dist = torch.norm(f[i] - f[j], p=2)
            u_dist = torch.norm(u_plus[i] - u_plus[j], p=2)
            total_edge_loss += torch.abs(f_dist - u_dist)
    return total_edge_loss / (n * (n - 1) / 2)

def average_features_by_label(z_, tgt_tra_bth, num_labels):
    label_to_features = defaultdict(list)

    for i, label in enumerate(tgt_tra_bth):
        label_to_features[int(label.item())].append(z_[i])

    averaged_features = []
    for label in range(num_labels):
        features_for_label = label_to_features[label]
        if features_for_label:
            averaged_features.append(torch.mean(torch.stack(features_for_label), dim=0))

    return torch.stack(averaged_features)

def generate_u_plus_u_minus(prox_z, prox_y, lambda_val):
    u_plus = []
    u_minus = []

    for label in prox_y:
        u1 = prox_z[label]
        u_plus.append(u1)

        for other_label in range(len(prox_z)):
            if other_label != label:
                u2 = prox_z[other_label]
                u_neg = negative_interpolation(u1, u2, lambda_val)
                u_minus.append(u_neg)

    u_plus = torch.stack(u_plus, dim=0)
    u_minus = torch.stack(u_minus, dim=0)
    return u_plus, u_minus

def train(net, trainloader, epochs, device,
          optimizer_fe, distill_optimizer,
          distill_loss_fn, distill_epochs, prox_z, prox_y, lambda_val=0.1):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    net.train()
    for epoch in range(distill_epochs):
        for _, (data_tra_bth, tgt_tra_bth) in enumerate(trainloader):
            batch_z = []
            for j in range(tgt_tra_bth.size(0)):
                batch_z.append(prox_z[int(tgt_tra_bth[j])])
            z = torch.stack(batch_z, dim=0)
            data_tra_bth, z = data_tra_bth.to(device), z.to(device)
            z_ = net.feature_extractor(data_tra_bth)

            u_plus, u_minus = generate_u_plus_u_minus(prox_z, prox_y, lambda_val)

            u_plus = u_plus.to(device)
            u_minus = u_minus.to(device)

            averaged_features = average_features_by_label(z_, tgt_tra_bth, len(prox_z))

            distill_optimizer.zero_grad()

            angle_loss_value = angle_loss(averaged_features, u_plus)
            edge_loss_value = edge_loss(averaged_features, u_plus)
            distill_loss_value = distill_loss_fn(z_, z)
            total_loss_value = angle_loss_value + edge_loss_value + distill_loss_value

            total_loss_value.backward()
            distill_optimizer.step()

    net.train()
    for epoch in range(epochs):
        for _, (data_tra_bth, tgt_tra_bth) in enumerate(trainloader):
            data_tra_bth, tgt_tra_bth = data_tra_bth.to(device), tgt_tra_bth.to(device)

            output_tra_bth = net(data_tra_bth)
            loss = criterion(output_tra_bth, tgt_tra_bth.long())

            optimizer_fe.zero_grad()
            loss.backward()
            optimizer_fe.step()

    return None


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
        with torch.no_grad():
            output_tes_bth = net(data_tes_bth)
            loss += criterion(output_tes_bth, tgt_tes_bth.long()).item()

            probabilities = nn.functional.softmax(output_tes_bth, dim=-1).cpu().numpy()
            all_probabilities.append(probabilities)

            predictions = torch.max(output_tes_bth.data, 1)[1]
            correct += (predictions == tgt_tes_bth).sum().item()

            all_targets.extend(tgt_tes_bth.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    all_probabilities = np.vstack(all_probabilities)
    return loss, accuracy, all_targets, all_predictions, all_probabilities


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
