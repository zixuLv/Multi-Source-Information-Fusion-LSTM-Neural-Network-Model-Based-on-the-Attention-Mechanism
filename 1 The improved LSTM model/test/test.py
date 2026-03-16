# train.py

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

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Call at the beginning of the script

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataset_loader, net, loss_function, optimizer):
    net.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for btsz, (shuju_1, shuju_2, shuju_3, shuju_4, label) in enumerate(dataset_loader):
        # Data preprocessing
        n_shuju_1 = shuju_1.permute(2, 0, 1).float().to(device)
        n_shuju_2 = shuju_2.permute(2, 0, 1).float().to(device)
        n_shuju_3 = shuju_3.permute(2, 0, 1).float().to(device)
        n_shuju_4 = shuju_4.permute(2, 0, 1).float().to(device)
        label = label.long().to(device)  # Convert labels to integer type

        optimizer.zero_grad()

        # Initialize hidden states
        batch_size = n_shuju_1.size(1)
        h_states, c_states = net.init_hidden(batch_size)

        # Forward propagation
        outputs, _, _ = net(n_shuju_1, n_shuju_2, n_shuju_3, n_shuju_4, h_states, c_states)
        # outputs shape should be [batch_size, 2]

        # Compute loss
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute predictions
        preds = torch.argmax(outputs, dim=1)
        total_correct += preds.eq(label).sum().item()
        total_samples += label.size(0)

    avg_loss = total_loss / len(dataset_loader)
    avg_acc = total_correct / total_samples

    print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_acc:.4f}")
    return avg_acc, avg_loss

def test(full_data, net, loss_function, device):
    net.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    # Extract full dataset
    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(device)
    shuju_2 = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(device)
    shuju_3 = torch.tensor(full_data['shuju_3']).permute(2, 0, 1).float().to(device)
    shuju_4 = torch.tensor(full_data['shuju_4']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)  # Convert labels to integer type

    # Initialize hidden states
    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    with torch.no_grad():
        # Forward propagation
        outputs, _, _ = net(shuju_1, shuju_2, shuju_3, shuju_4, h_states, c_states)

        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        # Compute predictions
        preds = torch.argmax(outputs, dim=1)
        if preds.shape != labels.shape:
            raise ValueError(f"Prediction and label size mismatch: preds={preds.shape}, labels={labels.shape}")
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds = preds.cpu().numpy().flatten()
        all_labels = labels.cpu().numpy().flatten()

    avg_loss = total_loss / 1  # Only one batch
    avg_acc = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}", '========================')

    return avg_acc, avg_loss, all_preds, all_labels

def extract_full_data(dataset):
    """
    Extract all data from Dataset into a full data dictionary.
    """
    shuju_1, shuju_2, shuju_3, shuju_4, labels = [], [], [], [], []

    # Iterate over the entire dataset
    for i in range(len(dataset)):
        s1, s2, s3, s4, label = dataset[i]
        shuju_1.append(s1)
        shuju_2.append(s2)
        shuju_3.append(s3)
        shuju_4.append(s4)
        labels.append(label)

    # Convert to tensors and return as dictionary
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
        # For multi-class classification, use CrossEntropy form
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of correct prediction
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":

    # ===============================
    # Load data (only once)
    # ===============================
    wenjian_name = glob.glob(r"dataset\dataset\*")
    total_shuju = U.chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    # ===============================
    # Automatically collect all Group configurations
    # ===============================
    groups = [
        getattr(C, name) for name in dir(C)
        if name.startswith("Group")
    ]

    print("The following groups will be executed:", [g.__name__ for g in groups])

    # ===============================
    # Run training for each Group
    # ===============================
    for group in groups:

        print("\n" + "=" * 60)
        print(f"Start running {group.__name__}")
        print("=" * 60)

        # ===============================
        # Data splitting
        # ===============================
        new_data_train, new_data_test = U.new_data_chuli(
            total_shuju,
            label_list,
            group,
            method='minmax',
            smoothing=False
        )

        dataset_train = U.Train_shuju(new_data_train)
        dataset_test = U.Test_shuju(new_data_test)

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
            'label': dataset_test.label,
        }

        # ===============================
        # Initialize model (reinitialize for each group)
        # ===============================
        net = Model1(C.canshu, device).to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=C.lr, weight_decay=5e-4)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.6,
            patience=5,
            verbose=True
        )

        # ===============================
        # Save directory (separate folder per group)
        # ===============================
        save_dir = os.path.join("ckpt", group.__name__)
        os.makedirs(save_dir, exist_ok=True)

        best_results = {
            'accuracy': 0,
            'preds': None,
            'labels': None,
            'epoch': 0,
            'ckpt_path': os.path.join(save_dir, "best_model.pth"),
            'confusion_matrix_path': os.path.join(save_dir, "best_confusion_matrix.png"),
            'results_file': os.path.join(save_dir, "best_test_results.xlsx")
        }

        max_test_acc = 0
        patience = 500
        no_improvement = 0

        # ===============================
        # Training loop
        # ===============================
        for epoch in range(C.epoch_zong):

            print(f"[{group.__name__}] Epoch {epoch + 1}/{C.epoch_zong}")

            train_acc, train_loss = train(dataset_train_loader, net, loss_function, optimizer)

            if (epoch + 1) % C.epoch_show == 0:

                test_acc, test_loss, all_preds, all_labels = test(
                    full_test_data,
                    net,
                    loss_function,
                    device
                )

                # Save best model
                if test_acc > best_results['accuracy']:
                    best_results['accuracy'] = test_acc
                    best_results['preds'] = all_preds
                    best_results['labels'] = all_labels
                    best_results['epoch'] = epoch + 1

                    torch.save(net.state_dict(), best_results['ckpt_path'])
                    print(f"✔ Best model saved: {test_acc:.4f}")

                # Early stopping tracking
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    no_improvement = 0
                else:
                    no_improvement += 1

                scheduler.step(test_acc)

            if no_improvement >= patience:
                print(f"{group.__name__} Early stopping triggered")
                break

        # ===============================
        # Save best results
        # ===============================
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
            plt.title(f'{group.__name__} Confusion Matrix (Epoch {best_results["epoch"]})')
            plt.savefig(best_results['confusion_matrix_path'])
            plt.close()

        print(f"{group.__name__} Training completed")
        print(f"Best test accuracy: {best_results['accuracy']:.4f}")