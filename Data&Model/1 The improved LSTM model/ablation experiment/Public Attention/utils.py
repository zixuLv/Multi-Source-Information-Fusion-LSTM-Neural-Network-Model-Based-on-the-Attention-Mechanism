import config as C
import glob, os, csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

x=2296
y=2429
z=x-500
# 数据归一化方法
def minmaxscaler(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-6)


def zscore_scaler(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + 1e-6)


def robust_scaler(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return (data - median) / (iqr + 1e-6)


# 数据平滑方法
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# 统一归一化接口
def normalize_data(data, method='minmax', smoothing=False, window_size=3):
    """
    数据归一化方法
    :param data: 输入数据
    :param method: 归一化方法 ('minmax', 'zscore', 'robust')
    :param smoothing: 是否启用平滑
    :param window_size: 滑动窗口大小（用于平滑）
    :return: 归一化后的数据
    """
    # print(data.shape)
    
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


# 初始化数据
def chushishuju(wenjian_name):
    total_shuju = {}
    for ii in wenjian_name:
        qq = ii.strip().split("\\")[-1].split("_")[0]
        # print(qq)
        # exit()
        if qq not in total_shuju:
            total_shuju[qq] = []
        ex_data = pd.read_csv(ii)
        for oo in range(ex_data.shape[0]):
            shuzhi = list(ex_data.iloc[oo])
            # print(len(shuzhi))
            
            shuzhi = [float(iii) for iii in shuzhi]
            total_shuju[qq].append(shuzhi)
    # exit()
    # print(total_shuju,'cccccccc')
    return total_shuju


def label_tal(dier):
    suoyin_data = pd.read_csv(dier)
    label_list = suoyin_data.values[0]
    label_list = list(label_list)
    return [float(iii) for iii in label_list]


# 数据预处理
def yuchuli(data, method='minmax', smoothing=False):
    datt = []
    for ooo in data:
        # print(ooo)
        ooo = normalize_data(ooo, method=method, smoothing=smoothing)
        # print(ooo)
        # exit()
        datt.append(ooo)
    return np.array(datt)


# 数据划分
def new_data_chuli(total_shuju, label_list, method='minmax', smoothing=False):
    new_data_train = {}
    new_data_test = {}
    new_data_tall = {}
    for i in total_shuju.keys():
        # print(i)
        if i not in new_data_train:
            new_data_train[i] = {"data": [], "label": []}
        if i not in new_data_test:
            new_data_test[i] = {"data": [], "label": []}
        if i not in new_data_tall:
            new_data_tall[i] = {"data": [], "label": []}

        shh = np.array(total_shuju[i])
        # print(len(shh))
        shuj = len(label_list) - C.canshu["sequence_length_num"]
        # print(shuj)
        
        for qq in range(shuj):
            shuhh = shh[:, qq:qq + C.canshu["sequence_length_num"]]
            base_shuhh =shuhh
            # print(len(base_shuhh[0]),'ccccccccccc',base_shuhh.shape)
            # print(len(base_shuhh))
            shuhh = yuchuli(shuhh, method=method, smoothing=smoothing)
            # print(len(shuhh[0]))
            # exit()
            labhh = label_list[qq + C.canshu["sequence_length_num"]] - label_list[qq + C.canshu["sequence_length_num"] - 1]
            # if len(base_shuhh)==8:
                # if label_list[qq + C.canshu["sequence_length_num"]]==65663.68987:
                    # print(base_shuhh[0])
                    # print(qq)
                    # exit()
            # exit()
            if labhh <= 0:
                new_data_tall[i]["label"].append(0)
            else:
                new_data_tall[i]["label"].append(1)
            new_data_tall[i]["data"].append(shuhh)
    # exit()
    # 数据集划分
    # x=0
    # y=1
    # for i in new_data_tall[i]["label"]:
        # if i==1:
            # x+=1
        # else:
            # y+=1
    # print(x,y)
    # exit()
    for i in total_shuju.keys():
        # x=531
        # y=718
        # print(new_data_tall[i]["data"][1].shape)
        # exit()
        # print(len(new_data_tall[i]["label"]))
        # exit()
        # new_data_train[i]["data"] = new_data_tall[i]["data"][:x] + new_data_tall[i]["data"][y:]
        # new_data_train[i]["label"] = new_data_tall[i]["label"][:x] + new_data_tall[i]["label"][y:]
        # new_data_test[i]["data"] = new_data_tall[i]["data"][x:y]
        # new_data_test[i]["label"] = new_data_tall[i]["label"][x:y]
        new_data_train[i]["data"] = new_data_tall[i]["data"][z:x]
        new_data_train[i]["label"] = new_data_tall[i]["label"][z:x]
        new_data_test[i]["data"] = new_data_tall[i]["data"][x:]
        new_data_test[i]["label"] = new_data_tall[i]["label"][x:]
    # print(new_data_train[])
    return new_data_train, new_data_test


# 训练数据类
class Train_shuju(Dataset):
    def __init__(self, new_data):
        dd = list(new_data.keys())
        # print(dd)
        self.shuju_1 = np.array(new_data[dd[0]]["data"])
        
        self.shuju_2 = np.array(new_data[dd[1]]["data"])
        self.shuju_3 = np.array(new_data[dd[2]]["data"])
        self.shuju_4 = np.array(new_data[dd[3]]["data"])
        self.label = np.array(new_data[dd[0]]["label"])
        # print(self.shuju_1.shape)
        # print(self.shuju_2.shape)
        # print(self.shuju_3.shape)
        # print(self.shuju_4.shape)
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


# 测试数据类
class Test_shuju(Dataset):
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
    # 数据加载
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju = chushishuju(wenjian_name)
    dier = "data\\transposed_suoyin.csv"
    label_list = label_tal(dier)

    # 数据预处理和划分
    new_data_train, new_data_test = new_data_chuli(total_shuju, label_list, method='minmax', smoothing=False)
    print("训练数据:", new_data_train.keys())
    print("训练数据:", len(new_data_train["Blockchain"]["label"]))
    # print(list(new_data_train.keys))
    dataset_train = Train_shuju(new_data_train) 
    # n_shuju_1 = shuju_1.permute(2, 0, 1).float().to(device)
    # n_shuju_2 = shuju_2.permute(2, 0, 1).float().to(device)
    # n_shuju_3 = shuju_3.permute(2, 0, 1).float().to(device)
    # n_shuju_4 = shuju_4.permute(2, 0, 1).float().to(device)
    # print("测试数据:", dataset_train.keys)
    