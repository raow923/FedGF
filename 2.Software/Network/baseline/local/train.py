#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leave-one-group-out training + per-subject evaluation & save .mat files.

Assumptions:
- client.txt: each line like: [35, 2, 41, 24, 16, 43]
- dataloader.py provides: get_train_data(..., path, ...), get_test_data(..., path, ...),
  train_BCIDataset(...), val_BCIDataset(...)
- net.py provides FeatureExtractor, Classifier, TwinBranchNets
- config.py provides necessary hyperparams:
    num_rounds, num_epo, num_data, bth_size, lr,
    f_down1,f_up1,... f_down4,f_up4, down_sample, fs, win_data, channel
  optionally 'seed' for reproducible sampling
"""

import os
import time
import ast
import random
import numpy as np
import scipy.io as sio

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import config as config
from net import TwinBranchNets, Classifier, FeatureExtractor
from dataloader import get_train_data, get_test_data, train_BCIDataset, val_BCIDataset
from tqdm import tqdm

# ---------------- Simple SubsetDataset ----------------
class SubsetDataset(Dataset):
    def __init__(self, data, labels, indices):
        self.data = [data[i] for i in indices]
        self.labels = [labels[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ---------------- Read client groups ----------------
def read_client_groups(txt_path):
    groups = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                arr = ast.literal_eval(line)
                if isinstance(arr, (list, tuple)):
                    groups.append([int(x) for x in arr])
            except Exception as e:
                print(f"[WARN] Failed to parse line: {line} -> {e}")
    return groups

# ---------------- Build data file paths ----------------
def load_data_paths_for_subjects(subjects):
    """
    Construct data paths from subject ids.
    If subject id < 10 -> zero pad (subj0X), else subjXX
    """
    paths = []
    for idx_sub in subjects:
        if idx_sub < 10:
            path_sub = f'C:\\Data\\sess01\\sess01_subj0{idx_sub}_EEG_SSVEP.mat'
        else:
            path_sub = f'C:\\Data\\sess01\\sess01_subj{idx_sub}_EEG_SSVEP.mat'
        # if idx_sub < 10:
        #     path_sub = f'/home/rao/Data/sess01/sess01_subj0{idx_sub}_EEG_SSVEP.mat'
        # else:
        #     path_sub = f'/home/rao/Data/sess01/sess01_subj{idx_sub}_EEG_SSVEP.mat'
        paths.append(path_sub)
    return paths

# ---------------- Evaluate single subject ----------------
def evaluate_subject(net, dataset_val, device, batch_size):
    """
    Evaluate one subject dataset and return accuracy, y_true, y_pred (numpy arrays)
    """
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1,
                            pin_memory=True, drop_last=False)
    net.eval()
    total = 0
    correct = 0
    y_true_batches = []
    y_pred_batches = []

    with torch.no_grad():
        for bth_val in val_loader:
            data_val_bth = bth_val[0].to(device)
            tgt_val_bth = bth_val[1].to(device)

            output = net(data_val_bth)
            _, pred = torch.max(output, dim=1)

            total += int(tgt_val_bth.size(0))
            correct += int((pred == tgt_val_bth).sum().item())

            y_true_batches.append(tgt_val_bth.detach().cpu().numpy())
            y_pred_batches.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)

    acc = (correct / total)
    return acc, y_true, y_pred

# ---------------- Main ----------------
def main():
    start_time_total = time.time()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # read client groups
    client_groups = read_client_groups('client.txt')
    if len(client_groups) == 0:
        raise RuntimeError("client.txt empty or unreadable. Please check file format.")
    print(f"[INFO] Found {len(client_groups)} groups in client.txt")

    results_dir = 'results_mat'

    # iterate groups: each group treated as test group (leave-one-group-out)
    for group_idx, test_group in enumerate(client_groups, start=1):
        print(f"\n[GROUP] Leave-out group {group_idx} as TEST: {test_group}")

        # training subjects = union of other groups
        train_subjects = []
        for gi, grp in enumerate(client_groups, start=1):
            if gi != group_idx:
                train_subjects.extend(grp)
        train_subjects = sorted(list(set(train_subjects)))

        # build file paths
        tra_paths = load_data_paths_for_subjects(train_subjects)
        test_paths = load_data_paths_for_subjects(test_group)

        feature_extractor = FeatureExtractor()
        classifier = Classifier()
        net = TwinBranchNets(feature_extractor, classifier)
        net.to(device)

        # config.num_rounds independent runs for this test group
        for r in range(1, config.num_rounds + 1):
            print(f"\n[RUN] Group {group_idx} - Round {r}/{config.num_rounds}")

            loss_f = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(net.parameters(), lr=getattr(config, 'lr', 1e-3), weight_decay=0.01)

            # ---------- Load training data from tra_paths ----------
            data_tra_list = []
            label_tra_list = []
            start_time_tra_list = []

            for path in tqdm(tra_paths, desc=f"Loading train data (group {group_idx} round {r})", unit='subj'):
                # get_train_data should accept path string
                mid1, mid2, mid3, mid4, label_tra, start_t = get_train_data(
                    config.f_down1, config.f_up1,
                    config.f_down2, config.f_up2,
                    config.f_down3, config.f_up3,
                    config.f_down4, config.f_up4,
                    path, config.down_sample, config.fs
                )
                data_tra = [mid1, mid2, mid3, mid4]
                data_tra_list.append(data_tra)
                label_tra_list.append(label_tra)
                start_time_tra_list.append(start_t)

            # combine all training samples
            all_data = []
            all_labels = []
            for data_tra, label_tra, st in zip(data_tra_list, label_tra_list, start_time_tra_list):
                dataset_tra = train_BCIDataset(int(config.num_data/32), data_tra, config.win_data,
                                               label_tra, st, config.down_sample, config.channel)
                for i in range(len(dataset_tra)):
                    d, l = dataset_tra[i]
                    all_data.append(d)
                    all_labels.append(l)

            # random sample config.num_data samples (or all if fewer)
            selected_indices = random.sample(range(len(all_data)), config.num_data)
            subset_dataset = SubsetDataset(all_data, all_labels, selected_indices)
            gen_tra = DataLoader(subset_dataset, shuffle=True, batch_size=config.bth_size,
                                 num_workers=1, pin_memory=True, drop_last=True)

            # ---------- Load validation/test datasets (per subject in test_group) ----------
            val_data_list = []
            val_label_list = []
            val_start_time_list = []
            for val_path in test_paths:
                m1, m2, m3, m4, label_val, st_val = get_test_data(
                    config.f_down1, config.f_up1,
                    config.f_down2, config.f_up2,
                    config.f_down3, config.f_up3,
                    config.f_down4, config.f_up4,
                    val_path, config.down_sample, config.fs
                )
                val_data_list.append([m1, m2, m3, m4])
                val_label_list.append(label_val)
                val_start_time_list.append(st_val)

            val_datasets = []
            for data_val, label_val, st_val in zip(val_data_list, val_label_list, val_start_time_list):
                ds = val_BCIDataset(config.num_data, data_val, config.win_data, label_val, st_val,
                                    config.down_sample, config.channel)
                val_datasets.append(ds)

            # ---------- Train ----------
            net.train()
            for epo in range(1, config.num_epo + 1):
                epoch_loss = 0.0
                iters = 0
                for bth in gen_tra:
                    data_b = bth[0].to(device)
                    tgt_b = bth[1].to(device)

                    optimizer.zero_grad()
                    out = net(data_b)
                    loss = loss_f(out, tgt_b.long())
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    iters += 1

                avg_loss = epoch_loss / max(1, iters)
                # print(f"[Train] Epoch {epo}/{config.num_epo} - Avg Loss: {avg_loss:.4f}")

            # ---------- Evaluate per-subject and save immediately ----------
            # val_datasets order must correspond to test_group order
            for subj_id, ds_val in zip(test_group, val_datasets):
                acc, y_t, y_p = evaluate_subject(net, ds_val, device, config.bth_size)

                # save .mat per subject per round
                save_name = os.path.join(results_dir, f"sub{subj_id}_round{r}.mat")
                sio.savemat(save_name, {
                    'subject_id': int(subj_id),
                    'round': int(r),
                    'accuracy': float(acc),
                    'y_true': np.array(y_t, dtype=int),
                    'y_pred': np.array(y_p, dtype=int)
                })
                print(f"[SAVE] {save_name} (sub {subj_id}, round {r}, acc={acc*100:.2f}%)")

            elapsed = time.time() - start_time_total
            days = int(elapsed // 86400)
            hours = int((elapsed % 86400) // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            print(f"[RUN DONE] Group {group_idx} Round {r} elapsed {days}d {hours}h {minutes}m {seconds}s")


if __name__ == '__main__':
    main()
