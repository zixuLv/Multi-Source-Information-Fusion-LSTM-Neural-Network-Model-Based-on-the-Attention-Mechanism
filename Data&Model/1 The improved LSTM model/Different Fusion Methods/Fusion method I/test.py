# test.py 

import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import config as C
import utils as U
from model.lstm import Model1

# 设置随机种子
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(full_data, net, loss_function, device):
    net.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 用于保存概率

    # 提取完整数据
    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(device)  # [seq_len, batch_size, input_size]
    shuju_2 = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(device)
    shuju_3 = torch.tensor(full_data['shuju_3']).permute(2, 0, 1).float().to(device)
    shuju_4 = torch.tensor(full_data['shuju_4']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)  # 确保标签为整数类型 [batch_size]

    # 初始化隐藏状态
    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    with torch.no_grad():
        # 前向传播
        outputs, _, _ = net(shuju_1, shuju_2, shuju_3, shuju_4, h_states, c_states)  # [batch_size, 2]
        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        # 计算预测概率
        probs = torch.softmax(outputs, dim=1)  # [batch_size, 2]
        all_probs = probs.cpu().numpy()

        # 选择预测类别
        preds = torch.argmax(probs, dim=1)  # [batch_size]
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds = preds.cpu().numpy()
        all_labels = labels.cpu().numpy()

    avg_loss = total_loss / 1  # 仅一个批次
    avg_acc = total_correct / total_samples
    print(f"测试损失: {avg_loss:.4f}, 测试准确率: {avg_acc:.4f}")

    return avg_acc, avg_loss, all_preds, all_labels, all_probs

if __name__ == "__main__":
    # 检查点路径
    checkpoint_path = "ckpt1/G4.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件未找到: {checkpoint_path}")

    # 数据加载
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju = U.chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    new_data_train, new_data_test = U.new_data_chuli(total_shuju, label_list, method='minmax', smoothing=False)

    dataset_test = U.Test_shuju(new_data_test)

    # 准备完整测试数据
    full_test_data = {
        'shuju_1': dataset_test.shuju_1,
        'shuju_2': dataset_test.shuju_2,
        'shuju_3': dataset_test.shuju_3,
        'shuju_4': dataset_test.shuju_4,
        'label': dataset_test.label,
    }

    # 初始化网络
    net = Model1(C.canshu, device).to(device)

    # 加载检查点
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    print(f"已加载检查点: {checkpoint_path}")

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 如果训练时使用了 FocalLoss，取消下面的注释，并注释上面的行
    # loss_function = FocalLoss(alpha=1, gamma=2, reduction='mean')

    # 运行测试
    test_acc, test_loss, all_preds, all_labels, all_probs = test(full_test_data, net, loss_function, device)

    # 保存测试结果
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存预测值、真实值和概率
    # 保存每个类别的概率
    prob_class_0 = all_probs[:, 0]
    prob_class_1 = all_probs[:, 1]

    results_df = pd.DataFrame({
        'True Label': all_labels,
        'Predicted Label': all_preds,
        'Probability Class 0': prob_class_0,
        'Probability Class 1': prob_class_1
    })
    results_file = os.path.join(results_dir, "test_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"测试结果已保存到: {results_file}")

    # 生成并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred down', 'Pred up'],
                yticklabels=['True down', 'True up'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"混淆矩阵已保存到: {confusion_matrix_path}")

    # 生成并保存详细的测试报告
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    metrics_df = pd.DataFrame({
        'Test Accuracy': [test_acc],
        'Test Loss': [test_loss],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    })
    metrics_file = os.path.join(results_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"测试指标已保存到: {metrics_file}")
    print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1 分数: {f1:.4f}")

    # 可选：绘制和保存预测分布图
    plt.figure(figsize=(10, 5))
    sns.countplot(x=all_labels, label="True", color='blue', alpha=0.6)
    sns.countplot(x=all_preds, label="Predicted", color='red', alpha=0.6)
    plt.legend(['True', 'Predicted'])
    plt.title('True vs Predicted Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    label_distribution_path = os.path.join(results_dir, "label_distribution.png")
    plt.savefig(label_distribution_path)
    plt.close()
    print(f"标签分布图已保存到: {label_distribution_path}")

    print("测试过程完成。") 
