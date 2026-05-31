"""
run_all_inference.py
────────────────────
Automatically traverses each subfolder, runs inference.py inside it,
then aggregates all results (Excel + confusion matrix images) into a
single output folder.
Directory structure example:
  root/
  ├── run_all_inference.py       ← place this script here
  ├── summary_results/           ← auto-created, all outputs go here
  ├── folder_A/
  │   └── inference.py
  ├── folder_B/
  │   └── inference.py
  └── ...
"""
import os
import sys
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from typing import List

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ROOT_DIR        = "."
INFERENCE_FILE  = "inference.py"
RESULT_FILE     = "inference_results.xlsx"
CM_FILE         = "inference_confusion_matrix.png"
SUMMARY_DIR     = "summary_results"
SUMMARY_EXCEL   = "all_results_summary.xlsx"
PYTHON_EXE      = sys.executable


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def find_inference_folders(root: str) -> List[Path]:
    """Find all subfolders containing inference.py, sorted by name."""
    root_path = Path(root).resolve()
    folders = sorted([
        p.parent
        for p in root_path.rglob(INFERENCE_FILE)
        if p.parent != root_path
    ])
    return folders


def run_inference(folder: Path) -> bool:
    """Run inference.py inside the given folder. Returns True on success."""
    script = folder / INFERENCE_FILE
    print(f"\n{'─'*60}")
    print(f"▶  Running: {script}")
    print(f"{'─'*60}")
    result = subprocess.run(
        [PYTHON_EXE, str(script)],
        cwd=str(folder),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"✗  [{folder.name}] failed (returncode={result.returncode})")
        return False
    print(f"✓  [{folder.name}] done")
    return True


def collect_and_copy(folders: List[Path], summary_dir: Path) -> pd.DataFrame:
    """
    For each successful folder:
      - Read inference_results.xlsx and append to master DataFrame
      - Copy confusion matrix image into summary_dir
    """
    dfs = []
    for folder in folders:
        # ── Excel ──────────────────────────────────────────────
        result_path = folder / RESULT_FILE
        if not result_path.exists():
            print(f"⚠  Excel not found, skipping: {result_path}")
        else:
            try:
                df = pd.read_excel(result_path)
                df.insert(0, "Source", folder.name)
                dfs.append(df)
                print(f"   Read: {result_path}  ({len(df)} rows)")
            except Exception as e:
                print(f"⚠  Failed to read [{folder.name}]: {e}")

        # ── Confusion matrix image ──────────────────────────────
        cm_src = folder / CM_FILE
        if cm_src.exists():
            cm_dst = summary_dir / f"{folder.name}_cm.png"
            shutil.copy2(cm_src, cm_dst)
            print(f"   Copied CM: {cm_dst.name}")
        else:
            print(f"⚠  CM image not found, skipping: {cm_src}")

    if not dfs:
        print("No Excel results to aggregate.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def build_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-folder accuracy summary table from True Label / Predicted Label.
    Computes:
      - Overall Accuracy : all samples correct / total
      - Accuracy of 1    : correctly predicted 1 / total true 1
      - Accuracy of 0    : correctly predicted 0 / total true 0
    """
    col_true = "True Label"
    col_pred = "Predicted Label"

    missing = [c for c in [col_true, col_pred] if c not in df.columns]
    if missing:
        print(f"⚠  Required columns not found: {missing}. Skipping accuracy summary.")
        return pd.DataFrame()

    rows = []
    for source, grp in df.groupby("Source"):
        true  = grp[col_true]
        pred  = grp[col_pred]

        total         = len(grp)
        overall_acc   = (true == pred).sum() / total

        pred_1        = pred == 1
        acc_1         = (true[pred_1] == 1).sum() / pred_1.sum() if pred_1.sum() > 0 else float("nan")

        pred_0        = pred == 0
        acc_0         = (true[pred_0] == 0).sum() / pred_0.sum() if pred_0.sum() > 0 else float("nan")

        rows.append({
            "Source":           source,
            "Total Samples":    total,
            "Overall Accuracy": f"{overall_acc:.4f}",
            "Accuracy of 1":    f"{acc_1:.4f}" if acc_1 == acc_1 else "N/A",
            "Accuracy of 0":    f"{acc_0:.4f}" if acc_0 == acc_0 else "N/A",
        })

    return pd.DataFrame(rows)


def save_summary(df: pd.DataFrame, output_path: Path):
    """Write aggregated DataFrame to Excel with multiple sheets."""
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:

        # Sheet 1: all sample details
        df.to_excel(writer, sheet_name="Details", index=False)

        # Sheet 2: per-folder accuracy summary (Overall / Class-1 / Class-0)
        acc_summary = build_accuracy_summary(df)
        if not acc_summary.empty:
            acc_summary.to_excel(writer, sheet_name="Accuracy Summary", index=False)
            print(f"\n{'='*60}")
            print("Accuracy Summary:")
            print(acc_summary.to_string(index=False))

        # Sheet 3: legacy Correct-column summary (kept for backward compatibility)
        if "Correct" in df.columns:
            legacy = (
                df.groupby("Source")["Correct"]
                .agg(Total="count", Correct="sum")
                .assign(Accuracy=lambda x: (x["Correct"] / x["Total"]).map("{:.4f}".format))
                .reset_index()
            )
            legacy.to_excel(writer, sheet_name="Correct Summary", index=False)

    print(f"\n✅  Summary Excel saved: {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. discover subfolders
    folders = find_inference_folders(ROOT_DIR)
    if not folders:
        print(f"No subfolders containing '{INFERENCE_FILE}' found under '{ROOT_DIR}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(folders)} subfolder(s):")
    for f in folders:
        print(f"  • {f.name}")

    # 2. create summary output folder
    summary_dir = Path(ROOT_DIR).resolve() / SUMMARY_DIR
    summary_dir.mkdir(exist_ok=True)
    print(f"\nSummary folder: {summary_dir}")

    # 3. run inference in each subfolder
    success_folders, failed_folders = [], []
    for folder in folders:
        ok = run_inference(folder)
        (success_folders if ok else failed_folders).append(folder)

    print(f"\nDone: {len(success_folders)} succeeded / {len(failed_folders)} failed")
    if failed_folders:
        print("Failed folders:")
        for f in failed_folders:
            print(f"  ✗  {f.name}")

    # 4. collect results and copy images into summary_dir
    print(f"\nAggregating results from {len(success_folders)} folder(s) ...")
    summary_df = collect_and_copy(success_folders, summary_dir)

    # 5. save aggregated Excel
    if not summary_df.empty:
        save_summary(summary_df, summary_dir / SUMMARY_EXCEL)
        print(f"\nTotal: {len(summary_df)} samples from "
              f"{summary_df['Source'].nunique()} folder(s).")
    else:
        print("Nothing to write.")