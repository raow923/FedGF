import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io as sio
from matplotlib import gridspec

# from matplotlib import rcParams
# rcParams['toolbar'] = 'None'  # 禁用工具栏

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

    # 使用 gridspec 创建一个图形，左边是混淆矩阵，右边是颜色条
    # plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 0.05])  # 设置宽度比例，右边的颜色条较窄

    # 绘制混淆矩阵
    ax_matrix = plt.subplot(gs[0])  # 左侧区域
    im = ax_matrix.matshow(cm_normalized, cmap=plt.cm.Blues)

    # 设置混淆矩阵的轴标签和字体大小
    ax_matrix.set_xticks(np.arange(len(class_names)))
    ax_matrix.set_yticks(np.arange(len(class_names)))

    # 调整标签位置，确保它们显示在底部和左侧
    ax_matrix.set_xticklabels(class_names, fontsize=10, ha='center')
    ax_matrix.set_yticklabels(class_names, fontsize=10, va='center')

    # 设置坐标轴标签，并设置字体样式
    ax_matrix.set_xlabel("Predicted Label", fontsize=20)
    ax_matrix.set_ylabel("True Label", fontsize=20)

    # 添加百分比标签
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = f"{cm_normalized[i, j]:.1f}%"  # 格式化为 1 位小数
            ax_matrix.text(j, i, percentage, ha="center", va="center",
                           color="black" if cm_normalized[i, j] < 50 else "white",
                           fontsize=6)

    # 调整坐标轴标签的位置
    ax_matrix.xaxis.set_ticks_position('bottom')
    ax_matrix.yaxis.set_ticks_position('left')

    # 创建颜色条并将其添加到右侧
    ax_cbar = plt.subplot(gs[1])  # 右侧区域
    cbar = plt.colorbar(im, cax=ax_cbar)  # 使用 cax 直接指定颜色条的轴

    # 设置颜色条的刻度和标签
    cbar.ax.tick_params(labelsize=15)  # 设置颜色条刻度的字体大小
    cbar.set_ticks([20, 40, 60, 80])  # 只显示 20, 40, 60, 80 这几个刻度
    cbar.set_ticklabels([f"{i / 100:.2f}" for i in range(20, 81, 20)])  # 格式化为小数显示

    # 自动调整布局，避免标签被遮挡
    plt.tight_layout()

    # 显示图形
    plt.show()

# ---------------------------
# 读取函数：按被试找到 accuracy 最大的 round，返回所有人的 best accuracy 与对应标签
# ---------------------------
def load_best_round_preds(num_rounds: int, num_subjects: int, dirpath: str = "results_mat"):
    """
    遍历 dirpath 中每个被试和每个 round 的 .mat 文件，找到每个被试在所有 round 中 accuracy 最大的那一轮，
    返回:
       - accuracies: list of float, length = num_subjects (best accuracy per subject)
       - y_trues_list: list of numpy arrays, 每个元素为该被试 best round 的 y_true
       - y_preds_list: list of numpy arrays, 对应的 y_pred
       - all_targets: 连接所有被试的 y_true（用于整体混淆矩阵）
       - all_predictions: 连接所有被试的 y_pred
       - best_rounds: list of round indices (as strings) chosen for each subject
    参数:
       num_rounds: 轮数总数（int）
       num_subjects: 被试数量（int），partition_id 假设为 0..num_subjects-1 或 1..num_subjects (取决你的命名)
       dirpath: client .mat 文件目录
    """
    accuracies = []
    y_trues_list = []
    y_preds_list = []
    best_rounds = []

    for subj_idx in range(1, num_subjects + 1):
        best_acc = -1.0
        best_y_true = np.array([], dtype=int)
        best_y_pred = np.array([], dtype=int)
        best_round = None

        # 搜索所有 round 的文件名（round 从 1 到 num_rounds）
        for r in range(1, num_rounds + 1):
            fpath = os.path.join(dirpath, f"sub{subj_idx}_round{r}.mat")

            mat = sio.loadmat(fpath)
            acc = float(mat.get("accuracy", np.nan))

            if acc > best_acc:
                best_acc = acc
                best_round = r
                best_y_true = np.array(mat["y_true"]).ravel().astype(int)
                best_y_pred = np.array(mat["y_pred"]).ravel().astype(int)

        accuracies.append(best_acc)
        y_trues_list.append(best_y_true)
        y_preds_list.append(best_y_pred)
        best_rounds.append(best_round)
        print(f"被试 {subj_idx} 的最佳 round={best_round} accuracy={best_acc:.4f}, y_true.shape={best_y_true.shape}")

    # 合并所有被试的标签用于整体混淆矩阵
    all_targets = np.concatenate([arr for arr in y_trues_list if arr.size > 0]) if any(arr.size > 0 for arr in y_trues_list) else np.array([], dtype=int)
    all_predictions = np.concatenate([arr for arr in y_preds_list if arr.size > 0]) if any(arr.size > 0 for arr in y_preds_list) else np.array([], dtype=int)

    acc = np.mean(accuracies)

    return acc, y_trues_list, y_preds_list, all_targets, all_predictions, best_rounds


if __name__ == '__main__':
    num_rounds = 50
    num_subjects = 54
    accuracies, y_trues_list, y_preds_list, all_targets, all_predictions, best_rounds = load_best_round_preds(num_rounds, num_subjects)
    print("平均准确率：", accuracies)

    class_names = [f"Class{i+1}" for i in range(4)]
    class_names = [f"{0.2 * i + 8:.1f}" for i in range(4)]

    plot_confusion_matrix(all_targets, all_predictions, class_names)


