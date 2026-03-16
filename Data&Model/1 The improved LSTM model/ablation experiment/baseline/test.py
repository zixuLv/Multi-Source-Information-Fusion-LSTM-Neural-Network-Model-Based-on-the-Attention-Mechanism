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

# =======================
# Set random seed (ensure reproducibility)
# =======================
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =======================
# Set computation device (GPU / CPU)
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# Test function
# =======================
def test(full_data, net, loss_function, device):
    net.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []  # Store predicted probabilities

    # -------- Extract full test data --------
    # Input format: [seq_len, batch_size, input_size]
    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(device)
    shuju_2 = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(device)
    shuju_3 = torch.tensor(full_data['shuju_3']).permute(2, 0, 1).float().to(device)
    shuju_4 = torch.tensor(full_data['shuju_4']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)  # Labels must be Long type

    # -------- Initialize LSTM hidden states --------
    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    with torch.no_grad():  # Disable gradient calculation for inference
        # -------- Forward pass --------
        outputs, _, _ = net(shuju_1, shuju_2, shuju_3, shuju_4, h_states, c_states)
        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        # -------- Compute prediction probabilities --------
        probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        all_probs = probs.cpu().numpy()

        # -------- Select predicted class --------
        preds = torch.argmax(probs, dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds = preds.cpu().numpy()
        all_labels = labels.cpu().numpy()

    avg_loss = total_loss  # Only one full batch
    avg_acc = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

    return avg_acc, avg_loss, all_preds, all_labels, all_probs


# =======================
# Main entry point
# =======================
if __name__ == "__main__":

    # -------- Check if checkpoint exists --------
    checkpoint_path = "ckpt/best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # -------- Load data --------
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju = U.chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    # Data preprocessing (normalization + split)
    new_data_train, new_data_test = U.new_data_chuli(
        total_shuju,
        label_list,
        method='minmax',
        smoothing=False
    )

    dataset_test = U.Test_shuju(new_data_test)

    # -------- Prepare full test data --------
    full_test_data = {
        'shuju_1': dataset_test.shuju_1,
        'shuju_2': dataset_test.shuju_2,
        'shuju_3': dataset_test.shuju_3,
        'shuju_4': dataset_test.shuju_4,
        'label': dataset_test.label,
    }

    # -------- Initialize model --------
    net = Model1(C.canshu, device).to(device)

    # -------- Load trained weights --------
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    print(f"Checkpoint loaded: {checkpoint_path}")

    # -------- Define loss function --------
    loss_function = nn.CrossEntropyLoss()
    # If FocalLoss was used during training, make sure to use the same here

    # -------- Run testing --------
    test_acc, test_loss, all_preds, all_labels, all_probs = test(
        full_test_data, net, loss_function, device
    )

    # =======================
    # Save test results
    # =======================
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)

    # Extract probabilities for each class
    prob_class_0 = all_probs[:, 0]
    prob_class_1 = all_probs[:, 1]

    # Save predictions, labels, and probabilities
    results_df = pd.DataFrame({
        'True Label': all_labels,
        'Predicted Label': all_preds,
        'Probability Class 0': prob_class_0,
        'Probability Class 1': prob_class_1
    })

    results_file = os.path.join(results_dir, "test_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Test results saved to: {results_file}")

    # =======================
    # Generate and save confusion matrix
    # =======================
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
    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    # =======================
    # Compute detailed evaluation metrics
    # =======================
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

    print(f"Test metrics saved to: {metrics_file}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # =======================
    # Plot label distribution comparison
    # =======================
    plt.figure(figsize=(10, 5))
    sns.countplot(x=all_labels, label="True", color='blue', alpha=0.6)
    sns.countplot(x=all_preds, label="Predicted", color='red', alpha=0.6)
    plt.legend(['True', 'Predicted'])
    plt.title('True vs Predicted Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')

    label_distribution_path = os.path.join(results_dir, "label_distribution.png")
    plt.savefig(label_distribution_path)
    plt.close()

    print(f"Label distribution plot saved to: {label_distribution_path}")
    print("Testing process completed.")