import config as C
import glob, os, csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

x = 2296
y = 2429
z = x - 500

# -----------------------------
# Data normalization methods
# -----------------------------

def minmaxscaler(data):
    """Min-Max normalization"""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-6)


def zscore_scaler(data):
    """Z-score normalization"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-6)


def robust_scaler(data):
    """Robust normalization (based on median and IQR)"""
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return (data - median) / (iqr + 1e-6)


# -----------------------------
# Data smoothing method
# -----------------------------

def moving_average(data, window_size=3):
    """Apply moving average smoothing"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# -----------------------------
# Unified normalization interface
# -----------------------------

def normalize_data(data, method='minmax', smoothing=False, window_size=3):
    """
    Normalize input data
    
    :param data: input data
    :param method: normalization method ('minmax', 'zscore', 'robust')
    :param smoothing: whether to apply smoothing
    :param window_size: sliding window size for smoothing
    :return: normalized data
    """

    if smoothing:
        data = moving_average(data, window_size=window_size)

    if method == 'minmax':
        return minmaxscaler(data)
    elif method == 'zscore':
        return zscore_scaler(data)
    elif method == 'robust':
        return robust_scaler(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# -----------------------------
# Initialize data
# -----------------------------

def chushishuju(wenjian_name):
    """Load raw data from CSV files"""
    total_shuju = {}
    for ii in wenjian_name:
        qq = ii.strip().split("\\")[-1].split("_")[0]

        if qq not in total_shuju:
            total_shuju[qq] = []

        ex_data = pd.read_csv(ii)

        for oo in range(ex_data.shape[0]):
            shuzhi = list(ex_data.iloc[oo])
            shuzhi = [float(iii) for iii in shuzhi]
            total_shuju[qq].append(shuzhi)

    return total_shuju


def label_tal(dier):
    """Load label data"""
    suoyin_data = pd.read_csv(dier)
    label_list = suoyin_data.values[0]
    label_list = list(label_list)
    return [float(iii) for iii in label_list]


# -----------------------------
# Data preprocessing
# -----------------------------

def yuchuli(data, method='minmax', smoothing=False):
    """Apply preprocessing (normalization + optional smoothing)"""
    datt = []
    for ooo in data:
        ooo = normalize_data(ooo, method=method, smoothing=smoothing)
        datt.append(ooo)
    return np.array(datt)


# -----------------------------
# Data splitting and windowing
# -----------------------------

def new_data_chuli(total_shuju, label_list, method='minmax', smoothing=False):
    """
    Create sliding window sequences and split into train/test sets
    """
    new_data_train = {}
    new_data_test = {}
    new_data_tall = {}

    for i in total_shuju.keys():

        if i not in new_data_train:
            new_data_train[i] = {"data": [], "label": []}
        if i not in new_data_test:
            new_data_test[i] = {"data": [], "label": []}
        if i not in new_data_tall:
            new_data_tall[i] = {"data": [], "label": []}

        shh = np.array(total_shuju[i])
        shuj = len(label_list) - C.canshu["sequence_length_num"]

        # Sliding window processing
        for qq in range(shuj):
            shuhh = shh[:, qq:qq + C.canshu["sequence_length_num"]]
            shuhh = yuchuli(shuhh, method=method, smoothing=smoothing)

            # Binary classification label (price up/down)
            labhh = label_list[qq + C.canshu["sequence_length_num"]] - \
                    label_list[qq + C.canshu["sequence_length_num"] - 1]

            if labhh <= 0:
                new_data_tall[i]["label"].append(0)
            else:
                new_data_tall[i]["label"].append(1)

            new_data_tall[i]["data"].append(shuhh)

    # Train-test split
    for i in total_shuju.keys():
        new_data_train[i]["data"] = new_data_tall[i]["data"][z:x]
        new_data_train[i]["label"] = new_data_tall[i]["label"][z:x]

        new_data_test[i]["data"] = new_data_tall[i]["data"][x:]
        new_data_test[i]["label"] = new_data_tall[i]["label"][x:]

    return new_data_train, new_data_test


# -----------------------------
# Training dataset class
# -----------------------------

class Train_shuju(Dataset):
    """PyTorch Dataset for training data"""

    def __init__(self, new_data):
        dd = list(new_data.keys())

        self.shuju_1 = np.array(new_data[dd[0]]["data"])
        self.shuju_2 = np.array(new_data[dd[1]]["data"])
        self.shuju_3 = np.array(new_data[dd[2]]["data"])
        self.shuju_4 = np.array(new_data[dd[3]]["data"])

        self.label = np.array(new_data[dd[0]]["label"])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return (
            self.shuju_1.astype(np.float32)[index],
            self.shuju_2.astype(np.float32)[index],
            self.shuju_3.astype(np.float32)[index],
            self.shuju_4.astype(np.float32)[index],
            self.label.astype(np.float32)[index],
        )


# -----------------------------
# Testing dataset class
# -----------------------------

class Test_shuju(Dataset):
    """PyTorch Dataset for testing data"""

    def __init__(self, new_data):
        dd = list(new_data.keys())

        self.shuju_1 = np.array(new_data[dd[0]]["data"])
        self.shuju_2 = np.array(new_data[dd[1]]["data"])
        self.shuju_3 = np.array(new_data[dd[2]]["data"])
        self.shuju_4 = np.array(new_data[dd[3]]["data"])

        self.label = np.array(new_data[dd[0]]["label"])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return (
            self.shuju_1.astype(np.float32)[index],
            self.shuju_2.astype(np.float32)[index],
            self.shuju_3.astype(np.float32)[index],
            self.shuju_4.astype(np.float32)[index],
            self.label.astype(np.float32)[index],
        )


if __name__ == "__main__":

    # Load dataset files
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju = chushishuju(wenjian_name)

    # Load labels
    dier = "data\\transposed_suoyin.csv"
    label_list = label_tal(dier)

    # Preprocess and split dataset
    new_data_train, new_data_test = new_data_chuli(
        total_shuju,
        label_list,
        method='minmax',
        smoothing=False
    )

    print("Training categories:", new_data_train.keys())
    print("Training sample size:", len(new_data_train["Blockchain"]["label"]))

    dataset_train = Train_shuju(new_data_train)