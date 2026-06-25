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

# def set_seed(seed=42):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset_loader, net, loss_function, optimizer):
    net.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for btsz, (shuju_1, shuju_2, shuju_3, shuju_4, label) in enumerate(dataset_loader):
        n_shuju_1 = shuju_1.permute(2, 0, 1).float().to(device)
        n_shuju_2 = shuju_2.permute(2, 0, 1).float().to(device)
        n_shuju_3 = shuju_3.permute(2, 0, 1).float().to(device)
        n_shuju_4 = shuju_4.permute(2, 0, 1).float().to(device)
        label = label.long().to(device)

        optimizer.zero_grad()

        batch_size = n_shuju_1.size(1)
        h_states, c_states = net.init_hidden(batch_size)

        outputs, _, _ = net(n_shuju_1, n_shuju_2, n_shuju_3, n_shuju_4, h_states, c_states)

        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        total_correct += preds.eq(label).sum().item()
        total_samples += label.size(0)

    avg_loss = total_loss / len(dataset_loader)
    avg_acc = total_correct / total_samples

    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")
    return avg_acc, avg_loss


def test(full_data, net, loss_function, device):
    net.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(device)
    shuju_2 = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(device)
    shuju_3 = torch.tensor(full_data['shuju_3']).permute(2, 0, 1).float().to(device)
    shuju_4 = torch.tensor(full_data['shuju_4']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)

    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    with torch.no_grad():
        outputs, _, _ = net(shuju_1, shuju_2, shuju_3, shuju_4, h_states, c_states)
        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        if preds.shape != labels.shape:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, labels={labels.shape}")
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds = preds.cpu().numpy().flatten()
        all_labels = labels.cpu().numpy().flatten()

    avg_loss = total_loss / 1
    avg_acc = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}", '========================')

    return avg_acc, avg_loss, all_preds, all_labels


def extract_full_data(dataset):
    shuju_1, shuju_2, shuju_3, shuju_4, labels = [], [], [], [], []

    for i in range(len(dataset)):
        s1, s2, s3, s4, label = dataset[i]
        shuju_1.append(s1)
        shuju_2.append(s2)
        shuju_3.append(s3)
        shuju_4.append(s4)
        labels.append(label)

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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # ── Data loading ────────────────────────────────────────────
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju = U.chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    new_data_train, new_data_test = U.new_data_chuli(
        total_shuju, label_list, method='minmax', smoothing=False
    )

    dataset_train = U.Train_shuju(new_data_train)
    dataset_test  = U.Test_shuju(new_data_test)

    dataset_train_loader = DataLoader(
        dataset_train,
        shuffle=True,
        num_workers=0,
        batch_size=C.canshu['batch_size_num'],
        drop_last=True
    )

    full_test_data = {
        'shuju_1': dataset_test.shuju_1,
        'shuju_2': dataset_test.shuju_2,
        'shuju_3': dataset_test.shuju_3,
        'shuju_4': dataset_test.shuju_4,
        'label'  : dataset_test.label,
    }

    # ── Model / loss / optimizer ─────────────────────────────────
    net = Model1(C.canshu, device).to(device)

    loss_function = nn.CrossEntropyLoss()
    # loss_function = FocalLoss(alpha=1, gamma=2, reduction='mean')

    optimizer = optim.Adam(net.parameters(), lr=C.lr, weight_decay=5e-4)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.6,
        patience=5,
        verbose=True
    )

    h_states, c_states = net.init_hidden(C.canshu['batch_size_num'])
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    # ── Training state ───────────────────────────────────────────
    train_acc_list,  test_acc_list  = [], []
    train_loss_list, test_loss_list = [], []
    max_test_acc  = 0
    no_improvement = 0
    patience       = 500

    os.makedirs("ckpt", exist_ok=True)

    best_results = {
        'accuracy'            : 0,
        'preds'               : None,
        'labels'              : None,
        'epoch'               : 0,
        'ckpt_path'           : "ckpt/best_model.pth",
        'confusion_matrix_path': "best_confusion_matrix.png",
        'results_file'        : "best_test_results.xlsx"
    }

    # ── Training loop ────────────────────────────────────────────
    for epoch in range(C.epoch_zong):
        print(f"Epoch {epoch + 1}/{C.epoch_zong}:")
        train_acc, train_loss = train(dataset_train_loader, net, loss_function, optimizer)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        if (epoch + 1) % C.epoch_show == 0:
            test_acc, test_loss, all_preds, all_labels = test(
                full_test_data, net, loss_function, device
            )
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

            # ── Save a checkpoint for every test ────────────────
            epoch_ckpt = f"ckpt/model_epoch{epoch + 1}_acc{test_acc:.4f}.pth"
            torch.save(net.state_dict(), epoch_ckpt)
            print(f"Checkpoint saved: {epoch_ckpt}")
            # ────────────────────────────────────────────────────

            # ── Keep track of the best model ────────────────────
            if test_acc > best_results['accuracy']:
                best_results['accuracy'] = test_acc
                best_results['preds']    = all_preds
                best_results['labels']   = all_labels
                best_results['epoch']    = epoch + 1
                torch.save(net.state_dict(), best_results['ckpt_path'])
                print(f"New best model saved, accuracy: {test_acc:.4f} at Epoch {epoch + 1}")

            if test_acc > max_test_acc:
                max_test_acc   = test_acc
                no_improvement = 0
            else:
                no_improvement += 1
            print('max_test_acc:', max_test_acc)

            scheduler.step(test_acc)

        if no_improvement >= patience:
            print(f"Early stopping triggered: no improvement for {patience} epochs.")
            break

    # ── Save best results ────────────────────────────────────────
    if best_results['accuracy'] > 0:
        results_df = pd.DataFrame({
            'True Label'     : best_results['labels'],
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

    print(f"Training complete.")
    print(f"Best test accuracy: {best_results['accuracy']:.4f} at Epoch {best_results['epoch']}")