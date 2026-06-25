# test_all.py
# 一键测试 ckpt1/ 下 G1~G5 所有权重，汇总结果到一张对比表格

import os
import glob
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import config as C
import utils as U
from model.lstm import Model1

# ───────────────────────────── 每个 G 段的数据划分参数 ─────────────────────────────
# 与 utils.py 中注释里的定义保持一致
G_SPLITS = {
    "G1": {"x": 2296,  "y": 2429,  "z": 1796},
    "G2": {"x": 1049, "y": 1183, "z": 549},
    "G3": {"x": 1184, "y": 1299, "z": 684},
    "G4": {"x": 1599, "y": 1734, "z": 1099},
    "G5": {"x": 1734, "y": 1867, "z": 1234},
}

# ───────────────────────────── 固定配置 ─────────────────────────────
CKPT_DIR     = "ckpt1"
DATA_GLOB    = r"yuchuli\dataset\*"
LABEL_FILE   = r"data\suoyin.csv"
RESULTS_DIR  = "test_results_all"
RESULTS_FILE = os.path.join(RESULTS_DIR, "summary.csv")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_test(full_data, net, loss_fn):
    """对一整批测试数据做一次前向推理，返回各项指标。"""
    net.eval()

    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(DEVICE)
    shuju_2 = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(DEVICE)
    shuju_3 = torch.tensor(full_data['shuju_3']).permute(2, 0, 1).float().to(DEVICE)
    shuju_4 = torch.tensor(full_data['shuju_4']).permute(2, 0, 1).float().to(DEVICE)
    labels  = torch.tensor(full_data['label']).long().to(DEVICE)

    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)
    h_states = [h.to(DEVICE) for h in h_states]
    c_states = [c.to(DEVICE) for c in c_states]

    with torch.no_grad():
        outputs, _, _ = net(shuju_1, shuju_2, shuju_3, shuju_4, h_states, c_states)
        loss  = loss_fn(outputs, labels).item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    preds_np  = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    acc  = (preds_np == labels_np).mean()
    prec = precision_score(labels_np, preds_np, zero_division=0)
    rec  = recall_score(labels_np, preds_np, zero_division=0)
    f1   = f1_score(labels_np, preds_np, zero_division=0)
    cm   = confusion_matrix(labels_np, preds_np)

    # 从混淆矩阵提取 TN/FP/FN/TP（二分类）
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "Loss":      round(loss,  4),
        "Accuracy":  round(acc,   4),
        "Precision": round(prec,  4),
        "Recall":    round(rec,   4),
        "F1":        round(f1,    4),
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
    }


def patch_splits(x, y, z):
    """临时修改 utils 模块中的全局划分变量，使其对应当前 G 段。"""
    U.x = x
    U.y = y
    U.z = z


def main():
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 一次性加载原始数据（所有 G 段共用同一份原始数据）
    print("加载原始数据...")
    wenjian_name = glob.glob(DATA_GLOB)
    total_shuju  = U.chushishuju(wenjian_name)
    label_list   = U.label_tal(LABEL_FILE)

    loss_fn = nn.CrossEntropyLoss()
    rows = []

    for gname, splits in G_SPLITS.items():
        ckpt_path = os.path.join(CKPT_DIR, f"{gname}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[跳过] 权重文件不存在: {ckpt_path}")
            continue

        print(f"\n{'='*40}")
        print(f"测试 {gname}  (x={splits['x']}, y={splits['y']}, z={splits['z']})")
        print(f"权重: {ckpt_path}")

        # 按当前 G 段的划分参数切分数据
        patch_splits(splits["x"], splits["y"], splits["z"])
        _, new_data_test = U.new_data_chuli(
            total_shuju, label_list, method='minmax', smoothing=False
        )

        dataset_test = U.Test_shuju(new_data_test)
        full_test_data = {
            'shuju_1': dataset_test.shuju_1,
            'shuju_2': dataset_test.shuju_2,
            'shuju_3': dataset_test.shuju_3,
            'shuju_4': dataset_test.shuju_4,
            'label':   dataset_test.label,
        }

        # 初始化并加载模型
        net = Model1(C.canshu, DEVICE).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"已加载权重: {ckpt_path}")

        metrics = run_test(full_test_data, net, loss_fn)
        metrics["Group"] = gname
        rows.append(metrics)

        print(f"  Acc={metrics['Accuracy']}  Prec={metrics['Precision']}  "
              f"Rec={metrics['Recall']}  F1={metrics['F1']}  Loss={metrics['Loss']}")
        print(f"  TP={metrics['TP']}  TN={metrics['TN']}  "
              f"FP={metrics['FP']}  FN={metrics['FN']}")

    # 汇总表格
    if rows:
        cols = ["Group", "Accuracy", "Precision", "Recall", "F1", "Loss",
                "TP", "TN", "FP", "FN"]
        summary_df = pd.DataFrame(rows)[cols]
        summary_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n{'='*40}")
        print("汇总结果:")
        print(summary_df.to_string(index=False))
        print(f"\n已保存到: {RESULTS_FILE}")
    else:
        print("没有任何可测试的权重文件，请检查 ckpt1/ 目录。")


if __name__ == "__main__":
    main()