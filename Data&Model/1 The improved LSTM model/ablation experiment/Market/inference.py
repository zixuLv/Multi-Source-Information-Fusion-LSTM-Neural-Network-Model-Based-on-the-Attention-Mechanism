import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import config as C
import utils as U
from model.lstm import Model1

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CKPT_PATH    = "ckpt/best_model.pth"   # path to the best checkpoint saved during training
OUTPUT_EXCEL = "inference_results.xlsx"
OUTPUT_CM    = "inference_confusion_matrix.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Data loading (mirrors train.py)
# ─────────────────────────────────────────────
def load_test_data():
    wenjian_name = glob.glob(r"yuchuli\dataset\*")
    total_shuju  = U.chushishuju(wenjian_name)

    dier       = "data\\suoyin.csv"
    label_list = U.label_tal(dier)

    _, new_data_test = U.new_data_chuli(
        total_shuju, label_list, method='minmax', smoothing=False
    )

    dataset_test = U.Test_shuju(new_data_test)

    full_test_data = {
        'shuju_1': dataset_test.shuju_1,
        'shuju_2': dataset_test.shuju_2,
        'shuju_3': dataset_test.shuju_3,
        'shuju_4': dataset_test.shuju_4,
        'label'  : dataset_test.label,
    }
    return full_test_data


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def inference(full_data, net, device):
    net.eval()

    # use shuju_1 only, consistent with train.py
    shuju  = torch.tensor(full_data['shuju_2']).permute(2, 0, 1).float().to(device)
    labels = torch.tensor(full_data['label']).long().to(device)

    batch_size         = shuju.size(1)
    h_states, c_states = net.init_hidden(batch_size)

    with torch.no_grad():
        outputs, _, _ = net(shuju, h_states, c_states)
        preds         = torch.argmax(outputs, dim=1)

    preds_np  = preds.cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()

    accuracy = (preds_np == labels_np).mean()
    return preds_np, labels_np, accuracy, outputs.cpu().numpy()


# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────
def save_results(preds, labels, logits):
    # convert logits to softmax probabilities
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    df = pd.DataFrame({
        'True Label'     : labels,
        'Predicted Label': preds,
        'Prob_Class0'    : probs[:, 0],
        'Prob_Class1'    : probs[:, 1],
        'Correct'        : (preds == labels).astype(int),
    })
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Results saved to {OUTPUT_EXCEL}")


def save_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['True 0', 'True 1']
    )
    plt.title('Inference Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(OUTPUT_CM)
    plt.close()
    print(f"Confusion matrix saved to {OUTPUT_CM}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. load test data
    print("Loading test data ...")
    full_test_data = load_test_data()

    # 2. build model and load checkpoint
    print(f"Loading model weights from: {CKPT_PATH}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}. Run train.py first.")

    net = Model1(C.canshu, device).to(device)
    net.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    net.eval()
    print("Model loaded successfully.")

    # 3. run inference
    print("Running inference ...")
    preds, labels, accuracy, logits = inference(full_test_data, net, device)

    # 4. print metrics
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=['Class 0', 'Class 1']))

    # 5. save results
    save_results(preds, labels, logits)
    save_confusion_matrix(preds, labels)

    print("\nInference complete.")