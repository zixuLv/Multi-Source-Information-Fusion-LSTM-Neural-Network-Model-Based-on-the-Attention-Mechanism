# -*- coding: utf-8 -*-
import config as C
import glob, os, csv
import numpy as np
import pandas as pd


def chushishuju(wenjian_name):
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
    suoyin_data = pd.read_csv(dier)
    label_list = suoyin_data.values[0]
    label_list = list(label_list)
    label_list = [float(iii) for iii in label_list]
    return label_list


def minmaxscaler(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val + 1e-4)


def yuchuli(data):
    datt = []
    for ooo in data:
        ooo = minmaxscaler(ooo)
        datt.append(ooo)
    datt = np.array(datt)
    datt = datt * C.beishu
    return datt


def new_data_chuli(total_shuju, label_list):

    new_data_train = {}
    new_data_test = {}
    new_data_tall = {}

    for i in total_shuju.keys():
        if i not in new_data_train:
            new_data_train[i] = {"data": [], "label": []}

    for i in total_shuju.keys():
        if i not in new_data_test:
            new_data_test[i] = {"data": [], "label": []}

    for i in total_shuju.keys():
        if i not in new_data_tall:
            new_data_tall[i] = {"data": [], "label": []}

        shh = np.array(total_shuju[i])

        # Prepare labels and sliding window length
        shuj = len(label_list) - C.canshu["sequence_length_num"]

        for qq in range(shuj):

            # Extract sliding window
            shuhh = shh[:, qq:qq + C.canshu["sequence_length_num"]]

            # Normalize each feature
            ii = []
            for xx in shuhh:
                j = minmaxscaler(xx)
                ii.append(j)
            shuhh = np.array(ii)

            # Compute label (price difference)
            labhh = label_list[qq + C.canshu["sequence_length_num"]] - \
                    label_list[qq + C.canshu["sequence_length_num"] - 1]

            print(shuj, "+++++++", len(shuhh), "==========", qq)

            if labhh <= 0:
                new_data_tall[i]["label"].append(0)
            else:
                new_data_tall[i]["label"].append(1)

            new_data_tall[i]["data"].append(shuhh)

    # Check dataset consistency
    print(new_data_tall.keys(), 'Checking dataset consistency')

    if len(new_data_tall["Public sentiment.csv"]["label"]) == \
       len(new_data_tall["Macroeconomics.csv"]["label"]) == \
       len(new_data_tall["Cryptocurrency.csv"]["label"]) == \
       len(new_data_tall["Blockchain.csv"]["label"]):
        print("Dataset alignment confirmed")
    else:
        print("Dataset mismatch detected")
        exit()

    total_fenge = len(new_data_tall["Public sentiment.csv"]["label"])
    fenge = total_fenge - int(total_fenge / 10)

    for i in total_shuju.keys():
        new_data_train[i]["data"] = new_data_tall[i]["data"][1796:2296]
        new_data_train[i]["label"] = new_data_tall[i]["label"][1796:2296]

        new_data_test[i]["data"] = new_data_tall[i]["data"][2296:2492]
        new_data_test[i]["label"] = new_data_tall[i]["label"][2296:2492]

    return new_data_train, new_data_test


def zhuanhuan(shuju_1):
    arr3 = []
    for i in shuju_1:
        arr = []
        for y in range(len(i[0])):
            temp = []
            for j in range(len(i)):
                temp.append(i[j][y])
            arr.append(temp)
        arr3.append(arr)
    return arr3


def mat_traverse(mat):
    dd = []
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            dd.append(mat[i, j])
    return dd


def load_data_train(train_path, strim=-1):
    x = []
    y = []
    if strim == -1:
        for line in open(train_path, "r", encoding="utf-8"):
            if line == ' ':
                break
            x_list, label = line.strip().split("\t")
            x_list = [xx for xx in x_list]
            x.append(x_list)
            y.append(label)
        return x, y
    else:
        for line in open(train_path, "r", encoding="utf-8"):
            x_list, label = line.strip().split("\t")
            x_list = [xx for xx in x_list]
            x_list = x_list[:strim] if len(x_list) > strim else x_list + [-1] * (strim - len(x_list))
            x.append(x_list)
            y.append(label)
        return x, y


def load_data_test(test_path, strim=-1):
    x_test = []
    y_test = []
    if strim == -1:
        for line in open(test_path, "r", encoding="utf-8"):
            if line == ' ':
                break
            x_test_list, label = line.strip().split("\t")
            x_test_list = [xx_test for xx_test in x_test_list]
            x_test.append(x_test_list)
            y_test.append(label)
        return x_test, y_test
    else:
        for line in open(test_path, "r", encoding="utf-8"):
            x_test_list, label = line.strip().split("\t")
            x_test_list = [xx_test for xx_test in x_test_list.split(",")]
            x_test_list = x_test_list[:strim] if len(x_test_list) > strim else x_test_list + [-1] * (strim - len(x_test_list))
            x_test.append(x_test_list)
            y_test.append(label)
        return x_test, y_test


def get_train_performance(y_predict, y, model, f):
    right = 0
    total = len(y)
    hx = {}
    a_a = 0
    a_b = 0
    b_a = 0
    b_b = 0

    for pre, rea in zip(y_predict, y):
        if rea == 1 and pre == rea:
            a_a += 1
            right += 1
        elif rea == 1 and pre != rea:
            a_b += 1
        elif rea == 0 and pre != rea:
            b_a += 1
        else:
            b_b += 1
            right += 1

    hx['a_a'] = a_a
    hx['a_b'] = a_b
    hx['b_a'] = b_a
    hx['b_b'] = b_b

    f.write("%s correct predictions: %s, accuracy: %s\n" % (model, right, right / total))
    return total, right


def get_test_performance(y_test_predict, y_test, model, f):
    total1 = len(y_test)
    right1 = 0
    hx = {}

    a_a = 0
    a_b = 0
    b_a = 0
    b_b = 0

    for pre1, rea1 in zip(y_test_predict, y_test):
        if rea1 == 1 and pre1 == rea1:
            a_a += 1
            right1 += 1
        elif rea1 == 1 and pre1 != rea1:
            a_b += 1
        elif rea1 == 0 and pre1 != rea1:
            b_a += 1
        else:
            b_b += 1
            right1 += 1

    hx['a_a'] = a_a
    hx['a_b'] = a_b
    hx['b_a'] = b_a
    hx['b_b'] = b_b

    precision_1 = a_a / (a_a + b_a) if (a_a + b_a) > 0 else 0
    recall_1 = a_a / (a_a + a_b) if (a_a + a_b) > 0 else 0
    precision_0 = b_b / (b_b + a_b) if (b_b + a_b) > 0 else 0
    recall_0 = b_b / (b_b + b_a) if (b_b + b_a) > 0 else 0

    f.write("%s correct predictions: %s, accuracy: %s\n" % (model, right1, right1 / total1))
    f.write("Class 1 precision: %s, recall: %s\n" % (precision_1, recall_1))
    f.write("Class 0 precision: %s, recall: %s\n" % (precision_0, recall_0))

    return total1, right1, hx, precision_1, recall_1, precision_0, recall_0


def main():
    wenjian_name = glob.glob(r"dataset\dataset\*")
    total_shuju = chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = label_tal(dier)

    new_data_train, new_data_test = new_data_chuli(total_shuju, label_list)

    dd = list(new_data_train.keys())
    shuju_1 = np.array(new_data_train[dd[0]]["data"])
    shuju_2 = np.array(new_data_train[dd[1]]["data"])
    shuju_3 = np.array(new_data_train[dd[2]]["data"])
    shuju_4 = np.array(new_data_train[dd[3]]["data"])
    label_train = np.array(new_data_train[dd[0]]["label"])

    arr3 = zhuanhuan(shuju_1)
    arr4 = zhuanhuan(shuju_2)
    arr5 = zhuanhuan(shuju_3)
    arr6 = zhuanhuan(shuju_4)

    ml_train = []
    for i in range(len(arr3)):
        aa = np.hstack((arr3[i], arr4[i]))
        bb = np.hstack((aa, arr5[i]))
        cc = np.hstack((bb, arr6[i]))
        dd = mat_traverse(cc)
        ml_train.append(dd)

    dd = list(new_data_test.keys())
    shuju_1 = np.array(new_data_test[dd[0]]["data"])
    shuju_2 = np.array(new_data_test[dd[1]]["data"])
    shuju_3 = np.array(new_data_test[dd[2]]["data"])
    shuju_4 = np.array(new_data_test[dd[3]]["data"])
    label_test = np.array(new_data_test[dd[0]]["label"])

    arr3 = zhuanhuan(shuju_1)
    arr4 = zhuanhuan(shuju_2)
    arr5 = zhuanhuan(shuju_3)
    arr6 = zhuanhuan(shuju_4)

    ml_test = []
    for i in range(len(arr3)):
        aa = np.hstack((arr3[i], arr4[i]))
        bb = np.hstack((aa, arr5[i]))
        cc = np.hstack((bb, arr6[i]))
        dd = mat_traverse(cc)
        ml_test.append(dd)

    return ml_train, label_train, ml_test, label_test


def show(list_1):
    plt.plot(list)
    plt.ylabel("label")
    plt.xticks(range(len(list_1)), rotation=0)
    plt.legend(["train"])
    plt.title(name)
    plt.savefig(name + ".png")
    plt.close()


if __name__ == "__main__":
    a, b, _, _ = main()