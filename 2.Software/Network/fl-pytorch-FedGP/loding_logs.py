import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io as sio
from matplotlib import gridspec


plt.rcParams['font.family'] = 'Times New Roman'

def plot_confusion_matrix(all_targets, all_predictions, class_names):
    if len(all_targets) == 0 or len(all_predictions) == 0:
        print("无有效的目标或预测数据！无法绘制混淆矩阵。")
        return

    cm = confusion_matrix(all_targets, all_predictions)

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 0.05])

    ax_matrix = plt.subplot(gs[0])
    im = ax_matrix.matshow(cm_normalized, cmap=plt.cm.Blues)

    ax_matrix.set_xticks(np.arange(len(class_names)))
    ax_matrix.set_yticks(np.arange(len(class_names)))

    ax_matrix.set_xticklabels(class_names, fontsize=15, ha='center')
    ax_matrix.set_yticklabels(class_names, fontsize=15, va='center')

    ax_matrix.set_xlabel("Predicted Label", fontsize=20)
    ax_matrix.set_ylabel("True Label", fontsize=20)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = f"{cm_normalized[i, j]:.1f}%"
            ax_matrix.text(j, i, percentage, ha="center", va="center",
                           color="black" if cm_normalized[i, j] < 50 else "white",
                           fontsize=15)

    ax_matrix.xaxis.set_ticks_position('bottom')
    ax_matrix.yaxis.set_ticks_position('left')

    ax_cbar = plt.subplot(gs[1])
    cbar = plt.colorbar(im, cax=ax_cbar)

    cbar.ax.tick_params(labelsize=15)
    cbar.set_ticks([20, 40, 60, 80])
    cbar.set_ticklabels([f"{i / 100:.2f}" for i in range(20, 81, 20)])

    plt.tight_layout()

    plt.show()

def load_best_round_preds(num_rounds: int, num_subjects: int, dirpath: str = "client_preds"):
    accuracies = []
    y_trues_list = []
    y_preds_list = []
    best_rounds = []

    for subj_idx in range(num_subjects):
        pid = str(subj_idx)
        best_acc = -1.0
        best_y_true = np.array([], dtype=int)
        best_y_pred = np.array([], dtype=int)
        best_round = None

        for r in range(1, num_rounds + 1):
            fpath = os.path.join(dirpath, f"preds_part{pid}_round{r}.mat")

            mat = sio.loadmat(fpath)
            acc = float(mat.get("accuracy", np.nan))

            if acc > best_acc:
                best_acc = acc
                best_round = r
                y_t = mat.get("y_true", None)
                y_p = mat.get("y_pred", None)
                # ensure 1-D numpy arrays
                best_y_true = np.array(y_t).ravel().astype(int)
                best_y_pred = np.array(y_p).ravel().astype(int)

        accuracies.append(best_acc)
        y_trues_list.append(best_y_true)
        y_preds_list.append(best_y_pred)
        best_rounds.append(best_round)
        print(f"被试 {pid} 的最佳 round={best_round} accuracy={best_acc:.4f}, y_true.shape={best_y_true.shape}")

    all_targets = np.concatenate([arr for arr in y_trues_list if arr.size > 0]) if any(arr.size > 0 for arr in y_trues_list) else np.array([], dtype=int)
    all_predictions = np.concatenate([arr for arr in y_preds_list if arr.size > 0]) if any(arr.size > 0 for arr in y_preds_list) else np.array([], dtype=int)

    acc = np.mean(accuracies)
    acc_std = np.std(accuracies)
    acc_sem = acc_std / np.sqrt(num_subjects)


    return acc, y_trues_list, y_preds_list, all_targets, all_predictions, best_rounds


if __name__ == '__main__':
    num_rounds = 300
    num_subjects = 12
    accuracies, y_trues_list, y_preds_list, all_targets, all_predictions, best_rounds = load_best_round_preds(num_rounds, num_subjects)
    print("平均准确率：", accuracies)

    class_names = [f"Class{i+1}" for i in range(6)]

    plot_confusion_matrix(all_targets, all_predictions, class_names)



