"""FL-PyTorch: A Flower / PyTorch app."""

from flwr.common import NDArray, NDArrays, Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fl_pytorch.task import (FeatureExtractor, Classifier, TwinBranchNets, FeatureGenerator,
                             get_weights, set_weights)
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, FitRes, EvaluateRes, FitIns, EvaluateIns, Scalar
from flwr.server.client_manager import ClientProxy, ClientManager
import time
from scipy.io import savemat
import os
import numpy as np
import json
import random
import torch
import fl_pytorch.config as config
import pickle
from typing import Union
from functools import partial, reduce
from torch.optim import Adam
import torch.nn as nn
def Myaggregate_inplace(results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
    scaling_factors = [fit_res.num_examples / num_examples_total for _, fit_res in results]
    def _try_inplace(
            x: NDArray,
            y: Union[NDArray, float],
            np_binary_op: np.ufunc
    ) -> NDArray:
        return (np_binary_op(x, y, out=x)
                if np.can_cast(y, x.dtype, casting="same_kind")
                else np_binary_op(x, np.array(y, x.dtype), out=x))

    params = [_try_inplace(x, scaling_factors[0], np_binary_op=np.multiply)
              for x in parameters_to_ndarrays(results[0][1].parameters)]

    for i, (_, fit_res) in enumerate(results[1:], start=1):
        res = (_try_inplace(x, scaling_factors[i], np_binary_op=np.multiply)
               for x in parameters_to_ndarrays(fit_res.parameters))

        params = [reduce(partial(_try_inplace, np_binary_op=np.add), layer_updates)
                  for layer_updates in zip(params, res)]

    return params

def format_elapsed_time(start_time: float) -> str:

    elapsed_time = time.time() - start_time
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(days)}天 {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"


class FedGP(FedAvg):
    def __init__(self, fraction_fit=0.75, fraction_evaluate=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.train_client_ids = []
        self.test_client_ids = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.generator = FeatureGenerator()

        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        self.gen_optimizer = Adam(self.generator.parameters(), lr=config.lr, weight_decay=1e-3)


    def gen_prox_data(self):
        porx_z = [0.] * config.n_class
        porx_y = list(range(config.n_class))

        batch_labels = np.random.choice(config.n_class, config.num_data)
        y = torch.tensor(batch_labels, dtype=torch.int64)

        z = self.generator(y)

        for i in range(config.n_class):
            idx = torch.nonzero(y == i).view(-1)
            if len(idx) > 0:

                porx_z[i] += (z[idx].sum(dim=0) / len(idx))

        return torch.stack(porx_z, dim=0), torch.tensor(porx_y)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        prox_z, porx_y = self.gen_prox_data()

        prox_z = prox_z.tolist()
        porx_y = porx_y.tolist()

        prox_z_serialized = pickle.dumps(prox_z)
        porx_y_serialized = pickle.dumps(porx_y)


        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)


        config["prox_z"] = prox_z_serialized
        config["porx_y"] = porx_y_serialized

        fit_ins = FitIns(parameters, config)

        if server_round == 1:
            num_clients = client_manager.num_available()
            num_fit_clients = int(self.fraction_fit * num_clients)

            available_clients = client_manager.sample(num_clients=num_clients)

            random.shuffle(available_clients)
            self.train_client_ids = [client.cid for client in available_clients[:num_fit_clients]]
            self.test_client_ids = [client.cid for client in available_clients[num_fit_clients:]]

        training_clients = [client for client in client_manager.sample(num_clients=client_manager.num_available())
                            if client.cid in self.train_client_ids]

        return [(client, fit_ins) for client in training_clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        evaluate_clients = [client for client in client_manager.sample(num_clients=client_manager.num_available())
                            if client.cid in self.test_client_ids]

        return [(client, evaluate_ins) for client in evaluate_clients]

    def update_generator(self, model):
        feature_extractor = FeatureExtractor()
        classifier = Classifier()
        net = TwinBranchNets(feature_extractor, classifier)
        for epoch in range(100):
            batch_labels = np.random.choice(config.n_class, config.num_data)
            y = torch.tensor(batch_labels, dtype=torch.int64)

            z = self.generator(y)

            set_weights(net, model)

            classifier_params = {k: v for k, v in net.state_dict().items() if 'classifier' in k}
            classifier_params = {k.replace('classifier.', ''): v for k, v in classifier_params.items()}


            new_classifier = Classifier()
            new_classifier.load_state_dict(classifier_params)
            new_classifier.eval()

            y_ = new_classifier(z)
            cls_loss = self.loss_fn(y_, y)
            self.gen_optimizer.zero_grad()
            cls_loss.backward()
            self.gen_optimizer.step()

            _, predicted = torch.max(y_, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)


        return None

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        aggregated_ndarrays = Myaggregate_inplace(results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.update_generator(aggregated_ndarrays)


        return parameters_aggregated, {}

    def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        if results:
            total_loss = 0.0
            total_accuracy = 0.0
            total_examples = 0
            all_targets = []
            all_predictions = []
            all_probabilities = []
            client_accuracies = []

            for _, evaluate_res in results:
                total_loss += evaluate_res.loss * evaluate_res.num_examples
                total_accuracy += evaluate_res.metrics.get("accuracy", 0.0) * evaluate_res.num_examples
                total_examples += evaluate_res.num_examples

                client_accuracy = evaluate_res.metrics.get("accuracy", 0.0)
                client_accuracies.append(client_accuracy)

                client_targets_json = evaluate_res.metrics.get("targets", "[]")
                client_predictions_json = evaluate_res.metrics.get("predictions", "[]")
                client_probabilities_json = evaluate_res.metrics.get("probabilities", "[]")

                client_targets = json.loads(client_targets_json)
                client_predictions = json.loads(client_predictions_json)
                client_probabilities = json.loads(client_probabilities_json)

                all_targets.extend(client_targets)
                all_predictions.extend(client_predictions)
                all_probabilities.extend(client_probabilities)

            avg_loss = total_loss / total_examples
            avg_accuracy = total_accuracy / total_examples

            all_targets_array = np.array(all_targets)
            all_predictions_array = np.array(all_predictions)
            all_probabilities_array = np.array(all_probabilities, dtype=np.float32)
            print(f"Round {rnd} - 测试的平均损失: {avg_loss:.4f}, 测试的平均准确率: {avg_accuracy:.4f}")

            logs_dir = "logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
                print(f"创建了日志文件夹: {logs_dir}")

            results_to_save = {
                "round": rnd,
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "targets": all_targets_array,
                "predictions": all_predictions_array,
                "probabilities": all_probabilities_array,
                "client_accuracies": np.array(client_accuracies),
            }
            filename = os.path.join(logs_dir, f"evaluation_results_round_{rnd}.mat")
            savemat(filename, results_to_save)
            print(f"程序总运行时间到 Round {rnd}: {format_elapsed_time(program_start_time)}")
            return avg_loss, {"accuracy": avg_accuracy}

        return None


def server_fn(context: Context):

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    fraction_evaluate = 1 - fraction_fit
    feature_extractor = FeatureExtractor()
    classifier = Classifier()

    ndarrays = get_weights(TwinBranchNets(feature_extractor, classifier))

    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedGP(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        min_evaluate_clients=1,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)



time.sleep(3)

program_start_time = time.time()

app = ServerApp(server_fn=server_fn)
