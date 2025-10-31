"""FL-PyTorch: A Flower / PyTorch app."""
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import json
from fl_pytorch.task import (
    FeatureExtractor, Classifier, TwinBranchNets,
    load_data,
    get_weights,
    set_weights,
    train,
    test, train_Prox,
)
import os
import numpy as np
import scipy.io as sio

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.net = net
        self.trainloader = trainloader    # 训练数据
        self.valloader = valloader      # 验证数据
        self.local_epochs = local_epochs   # 本地训练的轮次数
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)    # 将模型移动到计算设备上。
        self.partition_id = partition_id

    # 客户端在联邦学习中进行本地训练时调用的函数
    # parameters 由服务器端传递过来的全局模型参数
    def fit(self, parameters, config):
        # print(f'客户端{self.partition_id}开始训练！')
        # 将全局模型的参数设置为当前客户端的模型权重。
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        # train_loss = train_Prox(self.net, self.trainloader, self.local_epochs, self.device)
        # print(f'客户端{self.partition_id}开始训练！')
        # 返回训练完成后的模型权重、训练样本数以及训练损失。
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        # print(f'客户端{self.partition_id}开始评估！')
        round_in_fold = config["round_in_fold"]
        set_weights(self.net, parameters)  # 设置模型权重
        # 获取 loss, accuracy, targets, predictions, probabilities
        loss, accuracy, targets, predictions, probabilities = test(self.net, self.valloader, self.device)
        # 保存到 mat 文件
        out_dir = "client_preds"
        os.makedirs(out_dir, exist_ok=True)
        pid_str = str(self.partition_id)
        round_str = str(round_in_fold)
        filename = f"preds_part{pid_str}_round{round_str}.mat"
        filepath = os.path.join(out_dir, filename)

        sio.savemat(
            filepath,
            {
                "y_true": np.asarray(targets),
                "y_pred": np.asarray(predictions),
                "accuracy": float(accuracy),
                "loss": float(loss),
            },
        )

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    feature_extractor = FeatureExtractor()  # 特征提取器
    classifier = Classifier()  # 分类器
    # 创建 TwinBranchNets 网络
    net = TwinBranchNets(feature_extractor, classifier)
    partition_id = context.node_config["partition-id"]   # 从上下文中获取当前客户端的分区 ID。
    local_epochs = context.run_config["local-epochs"]   # 从上下文中获取本地训练的轮次。
    trainloader, valloader = load_data(partition_id)
    # Define client behavior based on partition_id
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()
# Flower ClientApp
app = ClientApp(client_fn)

