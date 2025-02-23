import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(all_targets, all_predictions, class_names):
    """
    绘制归一化的混淆矩阵，并显示百分比，同时在右侧添加颜色条表示深度。

    参数:
        all_targets (array-like): 真实标签。
        all_predictions (array-like): 预测标签。
        class_names (list): 分类标签的名称。
    """
    if len(all_targets) == 0 or len(all_predictions) == 0:
        print("无有效的目标或预测数据！无法绘制混淆矩阵。")
        return

    # 生成混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)

    # 归一化处理为百分比
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100  # 按行归一化为百分比

    # 设置图形
    fig, ax = plt.subplots(figsize=(8, 8))
    # 绘制混淆矩阵，启用颜色条
    im = ax.matshow(cm_normalized, cmap=plt.cm.Blues)

    # 添加颜色条，调整颜色条的大小，使其和矩阵一样高
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=12)  # 设置颜色条刻度的字体大小

    # 设置颜色条的刻度范围和显示格式
    cbar.set_ticks([20, 40, 60, 80])  # 只显示 20, 40, 60, 80 这几个刻度
    cbar.set_ticklabels([f"{i / 100:.2f}" for i in range(20, 81, 20)])  # 格式化为小数显示

    # 添加百分比标签
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = f"{cm_normalized[i, j]:.1f}%"  # 格式化为 1 位小数
            plt.text(j, i, percentage, ha="center", va="center",
                     color="black" if cm_normalized[i, j] < 50 else "white")

    # 设置轴标签和标题
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # 调整布局
    plt.tight_layout()
    plt.show()

def read_and_plot_logs(logs_dir="logs", max_rounds=None):
    """
    读取日志文件并绘制训练曲线，同时返回最高准确率对应轮次的数据。

    参数:
        logs_dir (str): 存储 .mat 日志文件的目录。
        max_rounds (int): 要绘制的最大轮次，如果为 None，则绘制所有轮次。

    返回:
        tuple: (all_targets, all_predictions, all_probabilities)
            - all_targets: 最高准确率轮次的真实标签。
            - all_predictions: 最高准确率轮次的预测标签。
            - all_probabilities: 最高准确率轮次的预测概率，保持矩阵形式。
    """
    # 检查 logs 文件夹是否存在
    if not os.path.exists(logs_dir):
        print(f"日志文件夹 {logs_dir} 不存在！")
        return None, None, None

    # 获取 logs 文件夹中的所有 .mat 文件，并按文件名中轮次排序
    mat_files = sorted([f for f in os.listdir(logs_dir) if f.endswith(".mat")],
                       key=lambda x: int(x.split("_")[-1].split(".")[0]))  # 根据轮次数字排序
    if not mat_files:
        print(f"日志文件夹 {logs_dir} 中没有找到 .mat 文件！")
        return None, None, None

    # 初始化数据容器
    rounds = []
    avg_losses = []
    avg_accuracies = []
    targets_dict = {}  # 保存每一轮的真实标签
    predictions_dict = {}  # 保存每一轮的预测标签
    probabilities_dict = {}  # 保存每一轮的预测概率（保持矩阵形式）
    client_accuracies_dict = {}  # 保存每一轮每个客户端的准确率

    # 遍历文件并读取数据
    for mat_file in mat_files:
        file_path = os.path.join(logs_dir, mat_file)
        data = loadmat(file_path)
        round_num = int(data["round"][0][0])  # 提取轮次编号

        # 如果指定了最大轮次限制，跳过超出范围的文件
        if max_rounds is not None and round_num > max_rounds:
            break

        # 提取数据
        rounds.append(round_num)
        avg_losses.append(data["avg_loss"][0][0])
        avg_accuracies.append(data["avg_accuracy"][0][0])
        # 保存每一轮的真实标签和预测标签
        targets_dict[round_num] = data["targets"].flatten()
        predictions_dict[round_num] = data["predictions"].flatten()
        # 保持 probabilities 的原始形状
        probabilities_dict[round_num] = data["probabilities"]
        # 保存每一轮每个客户端的准确率
        client_accuracies_dict[round_num] = data["client_accuracies"].flatten()

    # 检查是否有有效数据
    if not rounds:
        print(f"没有找到符合条件的轮次数据 (max_rounds={max_rounds})！")
        return None, None, None

    # 计算平均准确率的最大值及其对应轮次
    max_accuracy = max(avg_accuracies)
    max_accuracy_round = rounds[avg_accuracies.index(max_accuracy)]
    # 打印最大准确率和对应的轮次
    print(f"最高平均准确率: {max_accuracy:.4f}，出现在轮次: {max_accuracy_round}")

    # 计算该轮次客户端准确率的标准差
    client_accuracies = client_accuracies_dict[max_accuracy_round]
    accuracy_stddev = np.std(client_accuracies)
    print(f"最高准确率轮次 ({max_accuracy_round}) 的客户端准确率标准差: {accuracy_stddev:.4f}")

    # 获取最高准确率轮次对应的真实标签和预测标签
    all_targets = targets_dict[max_accuracy_round]
    all_predictions = predictions_dict[max_accuracy_round]
    all_probabilities = probabilities_dict[max_accuracy_round]

    # 绘制曲线
    plt.figure(figsize=(10, 6))

    # 平均损失曲线
    # plt.subplot(2, 1, 1)
    # plt.plot(rounds, avg_losses, marker="o", label="Average Loss")
    # plt.title("Training Metrics Over Rounds")
    # plt.xlabel("Round")
    # plt.ylabel("Average Loss")
    # plt.grid(True)
    # plt.legend()
    #
    # # 平均准确率曲线
    # plt.subplot(2, 1, 2)
    plt.plot(rounds, avg_accuracies, marker="o", color="green", label="Average Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    plt.legend()

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

    # 返回所有数据（真实标签、预测标签和预测概率保持矩阵形式）
    return all_targets, all_predictions, all_probabilities

if __name__ == '__main__':
    # # 读取日志并返回最高准确率对应的数据
    all_targets, all_predictions, all_probabilities = read_and_plot_logs(max_rounds=300)
    class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]

    plot_confusion_matrix(all_targets, all_predictions, class_names)

