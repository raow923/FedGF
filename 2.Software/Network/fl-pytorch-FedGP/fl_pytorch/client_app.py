"""FL-PyTorch: A Flower / PyTorch app."""
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import json
from torch.optim import AdamW
from fl_pytorch.task import (FeatureExtractor, Classifier, TwinBranchNets, VanillaKDLoss, load_data,
                             get_weights, set_weights, train, test)#, unfreeze, freeze)
import pickle
import fl_pytorch.config as config
import os
import numpy as np
import scipy.io as sio

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.partition_id = partition_id

        self.optimizer_fe = AdamW(self.net.parameters(), lr=config.lr, weight_decay=1e-3)
        self.distill_optimizer = AdamW(self.net.feature_extractor.parameters(), lr=config.lr, weight_decay=1e-3)
        self.distill_epochs = self.local_epochs
        self.distill_loss_fn = VanillaKDLoss(temperature=3)


    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        prox_z_serialized = config["prox_z"]
        porx_y_serialized = config["porx_y"]

        prox_z = pickle.loads(prox_z_serialized)
        porx_y = pickle.loads(porx_y_serialized)

        prox_z = torch.tensor(prox_z)
        porx_y = torch.tensor(porx_y)

        train(self.net, self.trainloader, self.local_epochs, self.device,
              self.optimizer_fe, self.distill_optimizer,
              self.distill_loss_fn, self.distill_epochs, prox_z, porx_y)
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        round_in_fold = config["round_in_fold"]
        set_weights(self.net, parameters)
        loss, accuracy, targets, predictions, probabilities = test(self.net, self.valloader, self.device)

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

        print(f"client{pid_str} acc:{accuracy}")

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        targets_json = json.dumps(targets)
        predictions_json = json.dumps(predictions)
        probabilities_json = json.dumps(probabilities.tolist())

        return loss, len(self.valloader.dataset), {
            "accuracy": accuracy,
            "targets": targets_json,
            "predictions": predictions_json,
            "probabilities": probabilities_json,
        }

def client_fn(context: Context):
    feature_extractor = FeatureExtractor()
    classifier = Classifier()
    net = TwinBranchNets(feature_extractor, classifier)
    partition_id = context.node_config["partition-id"]
    local_epochs = context.run_config["local-epochs"]
    trainloader, valloader = load_data(partition_id)
    # Define client behavior based on partition_id
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()
# Flower ClientApp
app = ClientApp(client_fn)

