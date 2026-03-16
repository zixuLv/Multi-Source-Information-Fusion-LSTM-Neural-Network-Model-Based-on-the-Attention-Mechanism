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

# =======================
# Set random seed (for reproducibility)
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Fix convolution algorithm
    torch.backends.cudnn.benchmark = False     # Disable auto optimization

set_seed(42)  # Call at the beginning of the script

# =======================
# Set running device (GPU / CPU)
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Training function
# =======================
def train(dataset_loader, net, loss_function, optimizer):
    net.train()  # Set to training mode
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for btsz, (shuju_1, shuju_2, shuju_3, shuju_4, label) in enumerate(dataset_loader):

        # -------- Data preprocessing --------
        n_shuju_1 = shuju_1.permute(2, 0, 1).float().to(device)
        label = label.long().to(device)  # Convert labels to Long (required for classification)

        optimizer.zero_grad()  # Clear gradients

        # -------- Initialize LSTM hidden states --------
        batch_size = n_shuju_1.size(1)
        h_states, c_states = net.init_hidden(batch_size)

        # -------- Forward propagation --------
        outputs, _, _ = net(n_shuju_1, h_states, c_states)
        # outputs shape: [batch_size, num_classes]

        # -------- Compute loss --------
        loss = loss_function(outputs, label)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        total_loss += loss.item()

        # -------- Compute training accuracy --------
        preds = torch.argmax(outputs, dim=1)
        total_correct += preds.eq(label).sum().item()
        total_samples += label.size(0)

    avg_loss = total_loss / len(dataset_loader)
    avg_acc = total_correct / total_samples

    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")
    return avg_acc, avg_loss


# =======================
# Testing function (full batch evaluation)
# =======================
def test(full_data, net, loss_function, device):
    net.eval()  # Set to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    # -------- Build full test data --------
    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)

    # -------- Initialize hidden states --------
    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)

    with torch.no_grad():  # Disable gradient computation
        outputs, _, _ = net(shuju_1, h_states, c_states)

        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        # Check dimension consistency
        if preds.shape != labels.shape:
            raise ValueError(f"Prediction and label size mismatch: preds={preds.shape}, labels={labels.shape}")

        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds = preds.cpu().numpy().flatten()
        all_labels = labels.cpu().numpy().flatten()

    avg_loss = total_loss
    avg_acc = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}", '========================')

    return avg_acc, avg_loss, all_preds, all_labels


# =======================
# Extract full dataset into a single dictionary
# =======================
def extract_full_data(dataset):
    """
    Combine all samples from Dataset into a full data dictionary
    """
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


# =======================
# Focal Loss (for class imbalance)
# =======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weight factor
        self.gamma = gamma  # Hard sample focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct prediction
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =======================
# Main entry point
# =======================
if __name__ == "__main__":

    # -------- Load data --------
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju = U.chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    # Data normalization + train/test split
    new_data_train, new_data_test = U.new_data_chuli(
        total_shuju,
        label_list,
        method='minmax',
        smoothing=False
    )

    dataset_train = U.Train_shuju(new_data_train)
    dataset_test = U.Test_shuju(new_data_test)

    # -------- Build DataLoader --------
    dataset_train_loader = DataLoader(
        dataset_train,
        shuffle=True,
        num_workers=0,
        batch_size=C.canshu['batch_size_num'],
        drop_last=True
    )

    # Build full test data
    full_test_data = {
        'shuju_1': dataset_test.shuju_1,
        'shuju_2': dataset_test.shuju_2,
        'shuju_3': dataset_test.shuju_3,
        'shuju_4': dataset_test.shuju_4,
        'label': dataset_test.label,
    }

    # -------- Initialize model --------
    net = Model1(C.canshu, device).to(device)

    # -------- Loss function --------
    loss_function = nn.CrossEntropyLoss()
    # For class imbalance, you can use FocalLoss instead:
    # loss_function = FocalLoss(alpha=1, gamma=2, reduction='mean')

    # -------- Optimizer --------
    optimizer = optim.Adam(net.parameters(), lr=C.lr, weight_decay=5e-4)

    # -------- Learning rate scheduler --------
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',   # Monitor test accuracy
        factor=0.6,
        patience=5,
        verbose=True
    )

    # -------- Early stopping parameters --------
    patience = 500
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

    # =======================
    # Start training
    # =======================
    for epoch in range(C.epoch_zong):
        print(f"Epoch {epoch + 1}/{C.epoch_zong}:")

        train_acc, train_loss = train(dataset_train_loader, net, loss_function, optimizer)

        if (epoch + 1) % C.epoch_show == 0:

            test_acc, test_loss, all_preds, all_labels = test(
                full_test_data, net, loss_function, device
            )

            # Save best model
            if test_acc > best_results['accuracy']:
                best_results['accuracy'] = test_acc
                best_results['preds'] = all_preds
                best_results['labels'] = all_labels
                best_results['epoch'] = epoch + 1
                torch.save(net.state_dict(), best_results['ckpt_path'])
                print(f"New best model saved, accuracy: {test_acc:.4f}")

            # Early stopping check
            if test_acc > best_results['accuracy']:
                no_improvement = 0
            else:
                no_improvement += 1

            scheduler.step(test_acc)

        if no_improvement >= patience:
            print("Early stopping triggered")
            break

    # -------- Save best results --------
    if best_results['accuracy'] > 0:
        results_df = pd.DataFrame({
            'True Label': best_results['labels'],
            'Predicted Label': best_results['preds']
        })
        results_df.to_excel(best_results['results_file'], index=False)

        cm = confusion_matrix(best_results['labels'], best_results['preds'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Epoch {best_results["epoch"]})')
        plt.savefig(best_results['confusion_matrix_path'])

    print("Training finished")
    print(f"Best Test Accuracy: {best_results['accuracy']:.4f}")