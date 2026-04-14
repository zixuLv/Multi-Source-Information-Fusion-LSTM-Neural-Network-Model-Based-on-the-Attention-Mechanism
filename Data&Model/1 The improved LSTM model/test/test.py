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

# Set random seed
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(full_data, net, loss_function, device):
    net.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []  # store prediction probabilities

    # Extract full dataset
    shuju_1 = torch.tensor(full_data['shuju_1']).permute(2, 0, 1).float().to(device)
    shuju_2 = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(device)
    shuju_3 = torch.tensor(full_data['shuju_3']).permute(2, 0, 1).float().to(device)
    shuju_4 = torch.tensor(full_data['shuju_4']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)

    # Initialize hidden states
    batch_size = shuju_1.size(1)
    h_states, c_states = net.init_hidden(batch_size)
    h_states = [h.to(device) for h in h_states]
    c_states = [c.to(device) for c in c_states]

    with torch.no_grad():
        # Forward pass
        outputs, _, _ = net(shuju_1, shuju_2, shuju_3, shuju_4, h_states, c_states)
        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        # Compute probabilities
        probs = torch.softmax(outputs, dim=1)
        all_probs = probs.cpu().numpy()

        # Predicted classes
        preds = torch.argmax(probs, dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        all_preds = preds.cpu().numpy()
        all_labels = labels.cpu().numpy()

    avg_loss = total_loss / 1
    avg_acc = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

    return avg_acc, avg_loss, all_preds, all_labels, all_probs

if __name__ == "__main__":
    # ======================
    # 1. Model checkpoint paths
    # ======================
    ckpt_root = r"E:\加密货币\final_code\test\ckpt"
    groups = [
        getattr(C, name) for name in dir(C)
        if name.startswith("Group")
    ]

    if len(groups) == 0:
        raise ValueError("No Group found!")

    print("Detected Groups:", groups)

    # ======================
    # 2. Load data (only once)
    # ======================
    wenjian_name = glob.glob(r"dataset\dataset\*")
    total_shuju = U.chushishuju(wenjian_name)

    dier = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    # ======================
    # 3. Output directory
    # ======================
    results_root = r"E:\加密货币\final_code\test\test_results"
    os.makedirs(results_root, exist_ok=True)

    # ======================
    # 4. Loss function
    # ======================
    loss_function = nn.CrossEntropyLoss()

    # ======================
    # 5. Iterate over each Group
    # ======================
    for group in groups:
        print("\n" + "=" * 60)
        print(f"Start running {group.__name__}")
        print("=" * 60)

        group_name = group.__name__
        group_path = os.path.join(ckpt_root, group_name)
        checkpoint_path = os.path.join(group_path, "best_model.pth")

        if not os.path.exists(checkpoint_path):
            print(f"Skip {group_name} (no model found)")
            continue

        # ======================
        # 5.1 Data split by group
        # ======================
        new_data_train, new_data_test = U.new_data_chuli(
            total_shuju,
            label_list,
            group,
            method='minmax',
            smoothing=False
        )

        dataset_test = U.Test_shuju(new_data_test)
                
        full_test_data = {
            'shuju_1': dataset_test.shuju_1,
            'shuju_2': dataset_test.shuju_2,
            'shuju_3': dataset_test.shuju_3,
            'shuju_4': dataset_test.shuju_4,
            'label': dataset_test.label,
        }

        # ======================
        # 5.2 Initialize model
        # ======================
        net = Model1(C.canshu, device).to(device)
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        net.eval()

        # ======================
        # 5.3 Run testing
        # ======================
        test_acc, test_loss, all_preds, all_labels, all_probs = test(
            full_test_data, net, loss_function, device
        )

        # ======================
        # 5.4 Output folder per group
        # ======================
        results_dir = os.path.join(results_root, group.__name__)
        os.makedirs(results_dir, exist_ok=True)

        # ======================
        # 5.5 Save CSV results
        # ======================
        prob_class_0 = all_probs[:, 0]
        prob_class_1 = all_probs[:, 1]

        results_df = pd.DataFrame({
            'True Label': all_labels,
            'Predicted Label': all_preds,
            'Probability Class 0': prob_class_0,
            'Probability Class 1': prob_class_1
        })
        results_df.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)

        # ======================
        # 5.6 Confusion matrix
        # ======================
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred down', 'Pred up'],
                    yticklabels=['True down', 'True up'])
        plt.title(f'Confusion Matrix - {group.__name__}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
        plt.close()

        # ======================
        # 5.7 Metrics
        # ======================
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        metrics_df = pd.DataFrame({
            'Test Accuracy': [test_acc],
            'Test Loss': [test_loss],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1]
        })
        metrics_df.to_csv(os.path.join(results_dir, "test_metrics.csv"), index=False)

        print(f"{group_name} testing completed ✔")

    print("\nAll model testing completed 🚀")