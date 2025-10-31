"""FL-PyTorch: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedAdam, FedMedian, FedTrimmedAvg, FedProx
from fl_pytorch.task import FeatureExtractor, Classifier, TwinBranchNets, get_weights
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, FitRes, EvaluateRes, FitIns, EvaluateIns, Scalar
from flwr.server.client_manager import ClientProxy, ClientManager
import time
import ast
from torch import nn

# 全局变量，保存固定的客户端顺序
_fixed_client_order = None


def format_elapsed_time(start_time: float) -> str:
    """格式化时间为 'X天 X小时 X分钟 X秒' 的字符串"""
    elapsed_time = time.time() - start_time
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(days)}天 {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"


def load_folds_from_txt(path: str) -> List[List[int]]:
    """从文件读取 folds，每行是一个列表格式"""
    folds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lst = ast.literal_eval(line)
                folds.append([int(x) for x in lst])
    return folds


def _get_proxies_by_ids(client_manager, ids: List[int]):
    """根据传入的整数 ids，返回固定顺序下的 client proxies"""
    global _fixed_client_order

    # 获取当前所有可用的 clients
    all_clients = client_manager.all()
    proxies = list(all_clients.values()) if isinstance(all_clients, dict) else list(all_clients)

    # 如果还没记录顺序，则初始化
    if _fixed_client_order is None:
        _fixed_client_order = list(proxies)

    # 根据固定顺序取 id 对应的 client
    selected = []
    for id_int in ids:
        selected.append(_fixed_client_order[id_int])

    return selected


class MyFedAvg(FedAvg):
    def __init__(
            self,
            *,
            model: nn.Module,
            rounds_per_fold: int = 3,  # 每个 fold（每次交叉验证的测试组）要运行的服务器轮数
            folds_file: str,  # 客户端分组文件
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.folds = load_folds_from_txt(folds_file)  # 加载客户端分组
        self.n_folds = len(self.folds)  # 客户端分组数量
        self.rounds_per_fold = rounds_per_fold

        # 构建每个 fold 的 train/test id 列表
        self.fold_train_ids: List[List[int]] = []
        self.fold_test_ids: List[List[int]] = []
        for test_idx in range(self.n_folds):
            train_ids = []
            for fi in range(self.n_folds):
                if fi != test_idx:
                    train_ids.extend([int(x) for x in self.folds[fi]])
            test_ids = [int(x) for x in self.folds[test_idx]]
            self.fold_train_ids.append(train_ids)
            self.fold_test_ids.append(test_ids)

        # 保存 initial model 参数（fold 切换时会重置为此初始状态）
        self._initial_nds = get_weights(self.model)

    def _round_to_fold(self, server_round: int) -> Tuple[int, int]:
        """
           将全局的 server 训练轮次 (server_round) 映射到交叉验证中的 (fold_idx, round_in_fold)。
        """
        zero_based = server_round - 1
        fold_idx = zero_based // self.rounds_per_fold
        round_in_fold = (zero_based % self.rounds_per_fold) + 1
        # 防止 fold_idx 超过 n_folds 范围（比如总轮数不是 rounds_per_fold 的整倍数）
        if fold_idx >= self.n_folds:
            fold_idx = self.n_folds - 1
        return int(fold_idx), int(round_in_fold)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)

        if round_in_fold == 1:
            parameters_to_send = ndarrays_to_parameters(self._initial_nds)
        else:
            parameters_to_send = parameters

        train_ids = self.fold_train_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, train_ids)

        # 构造 FitIns（参数与配置），你可以把更多超参放入 config 目前这个参数没有使用
        config = {"fold_idx": str(fold_idx), "round_in_fold": str(round_in_fold)}
        fit_ins = FitIns(parameters_to_send, config)
        return [(p, fit_ins) for p in selected_proxies]

    # 服务器端评估请求时给客户端下发的配置
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)
        test_ids = self.fold_test_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, test_ids)

        config = {"fold_idx": str(fold_idx), "round_in_fold": str((server_round - 1) % self.rounds_per_fold + 1)}
        eval_ins = EvaluateIns(parameters,config)
        return [(p, eval_ins) for p in selected_proxies]


class MyFedOpt(FedAdam):
    def __init__(
            self,
            *,
            model: nn.Module,
            rounds_per_fold: int = 3,  # 每个 fold（每次交叉验证的测试组）要运行的服务器轮数
            folds_file: str,  # 客户端分组文件
            **kwargs):
        self.model = model
        initial_parameters = ndarrays_to_parameters(get_weights(self.model))
        super().__init__(
            initial_parameters=initial_parameters,
            **kwargs)

        self.folds = load_folds_from_txt(folds_file)  # 加载客户端分组
        self.n_folds = len(self.folds)  # 客户端分组数量
        self.rounds_per_fold = rounds_per_fold

        # 构建每个 fold 的 train/test id 列表
        self.fold_train_ids: List[List[int]] = []
        self.fold_test_ids: List[List[int]] = []
        for test_idx in range(self.n_folds):
            train_ids = []
            for fi in range(self.n_folds):
                if fi != test_idx:
                    train_ids.extend([int(x) for x in self.folds[fi]])
            test_ids = [int(x) for x in self.folds[test_idx]]
            self.fold_train_ids.append(train_ids)
            self.fold_test_ids.append(test_ids)

        # 保存 initial model 参数（fold 切换时会重置为此初始状态）
        self._initial_nds = get_weights(self.model)

    def _round_to_fold(self, server_round: int) -> Tuple[int, int]:
        """
           将全局的 server 训练轮次 (server_round) 映射到交叉验证中的 (fold_idx, round_in_fold)。
        """
        zero_based = server_round - 1
        fold_idx = zero_based // self.rounds_per_fold
        round_in_fold = (zero_based % self.rounds_per_fold) + 1
        # 防止 fold_idx 超过 n_folds 范围（比如总轮数不是 rounds_per_fold 的整倍数）
        if fold_idx >= self.n_folds:
            fold_idx = self.n_folds - 1
        return int(fold_idx), int(round_in_fold)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)

        if round_in_fold == 1:
            parameters_to_send = ndarrays_to_parameters(self._initial_nds)
        else:
            parameters_to_send = parameters

        train_ids = self.fold_train_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, train_ids)

        # 构造 FitIns（参数与配置），你可以把更多超参放入 config 目前这个参数没有使用
        config = {"fold_idx": str(fold_idx), "round_in_fold": str(round_in_fold)}
        fit_ins = FitIns(parameters_to_send, config)
        return [(p, fit_ins) for p in selected_proxies]

    # 服务器端评估请求时给客户端下发的配置
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)
        test_ids = self.fold_test_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, test_ids)

        config = {"fold_idx": str(fold_idx), "round_in_fold": str((server_round - 1) % self.rounds_per_fold + 1)}
        eval_ins = EvaluateIns(parameters, config)
        return [(p, eval_ins) for p in selected_proxies]

class MyFedMedian(FedMedian):
    def __init__(
            self,
            *,
            model: nn.Module,
            rounds_per_fold: int = 3,  # 每个 fold（每次交叉验证的测试组）要运行的服务器轮数
            folds_file: str,  # 客户端分组文件
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.folds = load_folds_from_txt(folds_file)  # 加载客户端分组
        self.n_folds = len(self.folds)  # 客户端分组数量
        self.rounds_per_fold = rounds_per_fold

        # 构建每个 fold 的 train/test id 列表
        self.fold_train_ids: List[List[int]] = []
        self.fold_test_ids: List[List[int]] = []
        for test_idx in range(self.n_folds):
            train_ids = []
            for fi in range(self.n_folds):
                if fi != test_idx:
                    train_ids.extend([int(x) for x in self.folds[fi]])
            test_ids = [int(x) for x in self.folds[test_idx]]
            self.fold_train_ids.append(train_ids)
            self.fold_test_ids.append(test_ids)

        # 保存 initial model 参数（fold 切换时会重置为此初始状态）
        self._initial_nds = get_weights(self.model)

    def _round_to_fold(self, server_round: int) -> Tuple[int, int]:
        """
           将全局的 server 训练轮次 (server_round) 映射到交叉验证中的 (fold_idx, round_in_fold)。
        """
        zero_based = server_round - 1
        fold_idx = zero_based // self.rounds_per_fold
        round_in_fold = (zero_based % self.rounds_per_fold) + 1
        # 防止 fold_idx 超过 n_folds 范围（比如总轮数不是 rounds_per_fold 的整倍数）
        if fold_idx >= self.n_folds:
            fold_idx = self.n_folds - 1
        return int(fold_idx), int(round_in_fold)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)

        if round_in_fold == 1:
            parameters_to_send = ndarrays_to_parameters(self._initial_nds)
        else:
            parameters_to_send = parameters

        train_ids = self.fold_train_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, train_ids)

        # 构造 FitIns（参数与配置），你可以把更多超参放入 config 目前这个参数没有使用
        config = {"fold_idx": str(fold_idx), "round_in_fold": str(round_in_fold)}
        fit_ins = FitIns(parameters_to_send, config)
        return [(p, fit_ins) for p in selected_proxies]

    # 服务器端评估请求时给客户端下发的配置
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)
        test_ids = self.fold_test_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, test_ids)

        config = {"fold_idx": str(fold_idx), "round_in_fold": str((server_round - 1) % self.rounds_per_fold + 1)}
        eval_ins = EvaluateIns(parameters, config)
        return [(p, eval_ins) for p in selected_proxies]

class MyFedNova(FedAvg):
    def __init__(
            self,
            *,
            model: nn.Module,
            rounds_per_fold: int = 3,  # 每个 fold（每次交叉验证的测试组）要运行的服务器轮数
            folds_file: str,  # 客户端分组文件
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.folds = load_folds_from_txt(folds_file)  # 加载客户端分组
        self.n_folds = len(self.folds)  # 客户端分组数量
        self.rounds_per_fold = rounds_per_fold

        # 构建每个 fold 的 train/test id 列表
        self.fold_train_ids: List[List[int]] = []
        self.fold_test_ids: List[List[int]] = []
        for test_idx in range(self.n_folds):
            train_ids = []
            for fi in range(self.n_folds):
                if fi != test_idx:
                    train_ids.extend([int(x) for x in self.folds[fi]])
            test_ids = [int(x) for x in self.folds[test_idx]]
            self.fold_train_ids.append(train_ids)
            self.fold_test_ids.append(test_ids)

        # 保存 initial model 参数（fold 切换时会重置为此初始状态）
        self._initial_nds = get_weights(self.model)

    def _round_to_fold(self, server_round: int) -> Tuple[int, int]:
        """
           将全局的 server 训练轮次 (server_round) 映射到交叉验证中的 (fold_idx, round_in_fold)。
        """
        zero_based = server_round - 1
        fold_idx = zero_based // self.rounds_per_fold
        round_in_fold = (zero_based % self.rounds_per_fold) + 1
        # 防止 fold_idx 超过 n_folds 范围（比如总轮数不是 rounds_per_fold 的整倍数）
        if fold_idx >= self.n_folds:
            fold_idx = self.n_folds - 1
        return int(fold_idx), int(round_in_fold)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)

        if round_in_fold == 1:
            parameters_to_send = ndarrays_to_parameters(self._initial_nds)
        else:
            parameters_to_send = parameters

        train_ids = self.fold_train_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, train_ids)

        # 构造 FitIns（参数与配置），你可以把更多超参放入 config 目前这个参数没有使用
        config = {"fold_idx": str(fold_idx), "round_in_fold": str(round_in_fold)}
        fit_ins = FitIns(parameters_to_send, config)
        return [(p, fit_ins) for p in selected_proxies]

    # 服务器端评估请求时给客户端下发的配置
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)
        test_ids = self.fold_test_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, test_ids)

        config = {"fold_idx": str(fold_idx), "round_in_fold": str((server_round - 1) % self.rounds_per_fold + 1)}
        eval_ins = EvaluateIns(parameters,config)
        return [(p, eval_ins) for p in selected_proxies]

class MyFedTrimmedAvg(FedTrimmedAvg):
    def __init__(
            self,
            *,
            model: nn.Module,
            rounds_per_fold: int = 3,  # 每个 fold（每次交叉验证的测试组）要运行的服务器轮数
            folds_file: str,  # 客户端分组文件
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.folds = load_folds_from_txt(folds_file)  # 加载客户端分组
        self.n_folds = len(self.folds)  # 客户端分组数量
        self.rounds_per_fold = rounds_per_fold

        # 构建每个 fold 的 train/test id 列表
        self.fold_train_ids: List[List[int]] = []
        self.fold_test_ids: List[List[int]] = []
        for test_idx in range(self.n_folds):
            train_ids = []
            for fi in range(self.n_folds):
                if fi != test_idx:
                    train_ids.extend([int(x) for x in self.folds[fi]])
            test_ids = [int(x) for x in self.folds[test_idx]]
            self.fold_train_ids.append(train_ids)
            self.fold_test_ids.append(test_ids)

        # 保存 initial model 参数（fold 切换时会重置为此初始状态）
        self._initial_nds = get_weights(self.model)

    def _round_to_fold(self, server_round: int) -> Tuple[int, int]:
        """
           将全局的 server 训练轮次 (server_round) 映射到交叉验证中的 (fold_idx, round_in_fold)。
        """
        zero_based = server_round - 1
        fold_idx = zero_based // self.rounds_per_fold
        round_in_fold = (zero_based % self.rounds_per_fold) + 1
        # 防止 fold_idx 超过 n_folds 范围（比如总轮数不是 rounds_per_fold 的整倍数）
        if fold_idx >= self.n_folds:
            fold_idx = self.n_folds - 1
        return int(fold_idx), int(round_in_fold)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)

        if round_in_fold == 1:
            parameters_to_send = ndarrays_to_parameters(self._initial_nds)
        else:
            parameters_to_send = parameters

        train_ids = self.fold_train_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, train_ids)

        # 构造 FitIns（参数与配置），你可以把更多超参放入 config 目前这个参数没有使用
        config = {"fold_idx": str(fold_idx), "round_in_fold": str(round_in_fold)}
        fit_ins = FitIns(parameters_to_send, config)
        return [(p, fit_ins) for p in selected_proxies]

    # 服务器端评估请求时给客户端下发的配置
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)
        test_ids = self.fold_test_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, test_ids)

        config = {"fold_idx": str(fold_idx), "round_in_fold": str((server_round - 1) % self.rounds_per_fold + 1)}
        eval_ins = EvaluateIns(parameters, config)
        return [(p, eval_ins) for p in selected_proxies]

class MyFedProx(FedProx):
    def __init__(
            self,
            *,
            model: nn.Module,
            rounds_per_fold: int = 3,  # 每个 fold（每次交叉验证的测试组）要运行的服务器轮数
            folds_file: str,  # 客户端分组文件
            **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.folds = load_folds_from_txt(folds_file)  # 加载客户端分组
        self.n_folds = len(self.folds)  # 客户端分组数量
        self.rounds_per_fold = rounds_per_fold

        # 构建每个 fold 的 train/test id 列表
        self.fold_train_ids: List[List[int]] = []
        self.fold_test_ids: List[List[int]] = []
        for test_idx in range(self.n_folds):
            train_ids = []
            for fi in range(self.n_folds):
                if fi != test_idx:
                    train_ids.extend([int(x) for x in self.folds[fi]])
            test_ids = [int(x) for x in self.folds[test_idx]]
            self.fold_train_ids.append(train_ids)
            self.fold_test_ids.append(test_ids)

        # 保存 initial model 参数（fold 切换时会重置为此初始状态）
        self._initial_nds = get_weights(self.model)

    def _round_to_fold(self, server_round: int) -> Tuple[int, int]:
        """
           将全局的 server 训练轮次 (server_round) 映射到交叉验证中的 (fold_idx, round_in_fold)。
        """
        zero_based = server_round - 1
        fold_idx = zero_based // self.rounds_per_fold
        round_in_fold = (zero_based % self.rounds_per_fold) + 1
        # 防止 fold_idx 超过 n_folds 范围（比如总轮数不是 rounds_per_fold 的整倍数）
        if fold_idx >= self.n_folds:
            fold_idx = self.n_folds - 1
        return int(fold_idx), int(round_in_fold)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)

        if round_in_fold == 1:
            parameters_to_send = ndarrays_to_parameters(self._initial_nds)
        else:
            parameters_to_send = parameters

        train_ids = self.fold_train_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, train_ids)

        # 构造 FitIns（参数与配置），你可以把更多超参放入 config 目前这个参数没有使用
        config = {"fold_idx": str(fold_idx), "round_in_fold": str(round_in_fold)}
        fit_ins = FitIns(parameters_to_send, config)
        return [(p, fit_ins) for p in selected_proxies]

    # 服务器端评估请求时给客户端下发的配置
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        fold_idx, round_in_fold = self._round_to_fold(server_round)
        test_ids = self.fold_test_ids[fold_idx]
        selected_proxies = _get_proxies_by_ids(client_manager, test_ids)

        config = {"fold_idx": str(fold_idx), "round_in_fold": str((server_round - 1) % self.rounds_per_fold + 1)}
        eval_ins = EvaluateIns(parameters,config)
        return [(p, eval_ins) for p in selected_proxies]



def server_fn(context: Context):
    folds_file = "client.txt"
    rounds_per_fold = context.run_config["num-server-rounds"]  # 每个 fold 的服务器训练轮数（示例）

    n_folds = len(load_folds_from_txt(folds_file))  # 获取客户端分组数量
    total_rounds = rounds_per_fold * n_folds  # 交叉验证总轮次

    # 初始化模型参数
    feature_extractor = FeatureExtractor()  # 特征提取器
    classifier = Classifier()  # 分类器
    # 模型的参数（权重）
    model = TwinBranchNets(feature_extractor, classifier)

    # 使用自定义策略
    # strategy = MyFedAvg(
    #     model=model,
    #     folds_file=folds_file,
    #     rounds_per_fold=rounds_per_fold,
    # )

    # strategy = MyFedMedian(
    #     model=model,
    #     folds_file=folds_file,
    #     rounds_per_fold=rounds_per_fold,
    # )

    # strategy = MyFedTrimmedAvg(
    #     model=model,
    #     folds_file=folds_file,
    #     rounds_per_fold=rounds_per_fold,
    #     beta=0.15,
    # )

    # strategy = MyFedNova(
    #     model=model,
    #     folds_file=folds_file,
    #     rounds_per_fold=rounds_per_fold,
    # )

    # strategy = MyFedOpt(
    #     model=model,
    #     folds_file=folds_file,
    #     rounds_per_fold=rounds_per_fold,
    # )

    strategy = MyFedProx(
        model=model,
        folds_file=folds_file,
        rounds_per_fold=rounds_per_fold,
        proximal_mu=0.1,
    )

    config = ServerConfig(num_rounds=total_rounds)  # 配置服务器端的参数，联邦学习的轮次
    # 返回服务器的核心组件：包括所使用的联邦学习策略（FedAvg）和服务器配置（ServerConfig）。
    # ServerAppComponents 是 Flower 框架中的一个容器，用于打包服务器的所有配置和策略，交给 ServerApp 实例运行。
    return ServerAppComponents(strategy=strategy, config=config)

# 等待客户端连接
time.sleep(3)  # 设置等待时间，单位为秒
# 在脚本开始处记录程序的启动时间
program_start_time = time.time()

# Create ServerApp
app = ServerApp(server_fn=server_fn)
