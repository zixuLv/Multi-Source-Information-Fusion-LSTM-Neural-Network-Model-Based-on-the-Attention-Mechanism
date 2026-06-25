import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import config as C
import utils as U
from model.lstm import Model1
import torch.nn.functional as F
import random
import numpy as np

# 设置随机种子
# def set_seed(seed=42):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# set_seed(42)  # 在脚本开头调用

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataset_loader, net, loss_function, optimizer):
    net.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for btsz, (shuju_1, shuju_2, shuju_3, shuju_4, label) in enumerate(dataset_loader):
        # 数据预处理
        n_shuju_1 = shuju_1.permute(2, 0, 1).float().to(device)
        n_shuju_2 = shuju_2.permute(2, 0, 1).float().to(device)
        n_shuju_3 = shuju_3.permute(2, 0, 1).float().to(device)
        n_shuju_4 = shuju_4.permute(2, 0, 1).float().to(device)
        label = label.long().to(device)  # 转换为整数类型

        optimizer.zero_grad()

        # 初始化隐藏状态
        batch_size = n_shuju_1.size(1)
        h_states, c_states = net.init_hidden(batch_size)

        # 前向传播
        outputs, _, _ = net(n_shuju_1, n_shuju_2, n_shuju_3, n_shuju_4, h_states, c_states)
        # outputs 的形状应为 [batch_size, 2]

        # 计算损失
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算预测
        preds = torch.argmax(outputs, dim=1)
        total_correct += preds.eq(label).sum().item()
        total_samples += label.size(0)

    avg_loss = total_loss / len(dataset_loader)
    avg_acc = total_correct / total_samples

    print(f"训练损失: {avg_loss:.4f}, 训练准确率: {avg_acc:.4f}")
    return avg_acc, avg_loss

def test(full_data, net, loss_function, device):
    net.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    # 提取完整数据
    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(device)
    shuju_2 = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(device)
    shuju_3 = torch.tensor(full_data['shuju_3']).permute(2, 0, 1).float().to(device)
    shuju_4 = torch.tensor(full_data['shuju_4']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)  # 转换为整数类型

    # 初始化隐藏状态
    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    with torch.no_grad():
        # 前向传播
        outputs, _, _ = net(shuju_1, shuju_2, shuju_3, shuju_4, h_states, c_states)
        # print(outputs)
        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        # 计算预测
        preds = torch.argmax(outputs, dim=1)
        if preds.shape != labels.shape:
            raise ValueError(f"预测值和真实值尺寸不匹配: preds={preds.shape}, labels={labels.shape}")
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds = preds.cpu().numpy().flatten()
        all_labels = labels.cpu().numpy().flatten()

    avg_loss = total_loss / 1  # 仅一个批次
    avg_acc = total_correct / total_samples
    print(f"测试损失: {avg_loss:.4f}, 测试准确率: {avg_acc:.4f}", '========================')

    return avg_acc, avg_loss, all_preds, all_labels

def extract_full_data(dataset):
    """
    将 Dataset 中的所有数据提取为完整数据字典。
    """
    shuju_1, shuju_2, shuju_3, shuju_4, labels = [], [], [], [], []

    # 遍历整个 Dataset
    for i in range(len(dataset)):
        s1, s2, s3, s4, label = dataset[i]
        shuju_1.append(s1)
        shuju_2.append(s2)
        shuju_3.append(s3)
        shuju_4.append(s4)
        labels.append(label)

    # 转换为张量，并返回字典
    return {
        'shuju_1': torch.stack(shuju_1),
        'shuju_2': torch.stack(shuju_2),
        'shuju_3': torch.stack(shuju_3),
        'shuju_4': torch.stack(shuju_4),
        'label': torch.tensor(labels)
    }

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 对于多类别，使用 CrossEntropyLoss 的形式
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt 是预测正确的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    # 数据加载
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju = U.chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    new_data_train, new_data_test = U.new_data_chuli(total_shuju, label_list, method='minmax', smoothing=False)

    dataset_train = U.Train_shuju(new_data_train)
    dataset_test = U.Test_shuju(new_data_test)

    dataset_train_loader = DataLoader(
        dataset_train,
        shuffle=True,  # 训练集通常需要打乱
        num_workers=0,
        batch_size=C.canshu['batch_size_num'],
        drop_last=True
    )

    # 准备完整测试数据
    full_test_data ={
        'shuju_1': dataset_test.shuju_1,
        'shuju_2': dataset_test.shuju_2,
        'shuju_3': dataset_test.shuju_3,
        'shuju_4': dataset_test.shuju_4,
        'label': dataset_test.label,
    }

    # 初始化网络
    net = Model1(C.canshu, device).to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    # 如果要使用 FocalLoss，取消下面的注释，并注释上面的行
    # loss_function = FocalLoss(alpha=1, gamma=2, reduction='mean')  # 确保 FocalLoss 支持多类别

    optimizer = optim.Adam(net.parameters(), lr=C.lr, weight_decay=1e-4)

    # 自适应学习率
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控的是准确率
        factor=0.6,
        patience=5,
        verbose=True
    )

    # 初始化隐藏状态
    h_states, c_states = net.init_hidden(C.canshu['batch_size_num'])
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    # 训练与测试
    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [], []  # 新增列表
    max_test_acc = 0

    # 早停机制
    patience = 500  # 根据需要调整
    no_improvement = 0

    os.makedirs("ckpt", exist_ok=True)

    best_results = {
        'accuracy': 0,
        'preds': None,
        'labels': None,
        'epoch': 0,
        'ckpt_path': "ckpt/best_model.pth",
        'confusion_matrix_path': "best_confusion_matrix.png",
        'results_file': "best_test_results.xlsx"
    }

    for epoch in range(C.epoch_zong):
        print(f"Epoch {epoch + 1}/{C.epoch_zong}:")
        train_acc, train_loss = train(dataset_train_loader, net, loss_function, optimizer)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)  # 记录训练损失

        if (epoch + 1) % C.epoch_show == 0:
            test_acc, test_loss, all_preds, all_labels = test(full_test_data, net, loss_function, device)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)  # 记录测试损失

            # 每次测试都保留一份权重（带 epoch 编号和准确率）
            epoch_ckpt = f"ckpt/model_epoch{epoch + 1}_acc{test_acc:.4f}.pth"
            torch.save(net.state_dict(), epoch_ckpt)
            print(f"Checkpoint saved: {epoch_ckpt}")

            if test_acc > best_results['accuracy']:
                best_results['accuracy'] = test_acc
                best_results['preds'] = all_preds
                best_results['labels'] = all_labels
                best_results['epoch'] = epoch + 1
                # 保存最佳模型
                torch.save(net.state_dict(), best_results['ckpt_path'])
                print(f"新最佳模型已保存，准确率: {test_acc:.4f} at Epoch {epoch + 1}")

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                no_improvement = 0
            else:
                no_improvement += 1
            print('max_test_acc:', max_test_acc)

            # 更新学习率调度器
            scheduler.step(test_acc)

        if no_improvement >= patience:
            print(f"早停触发：在 {patience} 个 epoch 内测试准确率没有提升。")
            break
    '''
    # 保存准确率和损失到 CSV 文件
    pd.DataFrame({'Train Accuracy': train_acc_list, 'Train Loss': train_loss_list}).to_csv("train_metrics.csv", index=False)
    pd.DataFrame({'Test Accuracy': test_acc_list, 'Test Loss': test_loss_list}).to_csv("test_metrics.csv", index=False)
    
    # 绘制并保存损失曲线
    epochs_train = range(1, len(train_loss_list) + 1)
    epochs_test = range(1, len(test_loss_list) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_train, train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("train_loss_curve.png")
    plt.close()

    if test_loss_list:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_test, test_loss_list, label='Test Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig("test_loss_curve.png")
        plt.close()

    # 绘制并保存准确率曲线（可选）
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_train, train_acc_list, label='Train Accuracy')
    plt.plot(epochs_test, test_acc_list, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve.png")
    plt.close()
    plt.close()
    '''
    if best_results['accuracy'] > 0:
        results_df = pd.DataFrame({
            'True Label': best_results['labels'],
            'Predicted Label': best_results['preds']
        })
        results_df.to_excel(best_results['results_file'], index=False)

        cm = confusion_matrix(best_results['labels'], best_results['preds'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'])
        plt.title(f'Confusion Matrix (Epoch {best_results["epoch"]})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(best_results['confusion_matrix_path'])

    print(f"训练结束。")
    print(f"最高测试准确率: {best_results['accuracy']:.4f} 在 Epoch {best_results['epoch']}")