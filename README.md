# Multi-Source-Information-Fusion-LSTM-Neural-Network-Model-Based-on-the-Attention-Mechanism

[![Python 3.8.20](https://img.shields.io/badge/python-3.8.20-blue.svg)](https://www.python.org/downloads/release/python-3820/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains the complete reproducibility package for our research on multi-source information fusion using an improved LSTM neural network with attention mechanism for cryptocurrency price prediction and trading strategy generation.

**Assembled on:** March 2, 2026

---

## 👥 Authors

| Author | Responsibility | Contact |
|--------|---------------|---------|
| **Huicong Liu** | The improved LSTM model | lcong0308@gmail.com |
| **Yinghan Zhang** | Time series model | zhaoyinghan97@163.com |
| **Zixu Lv** | Asset pricing model | lvzixu666@163.com |

---

## 🖥️ Computing Environment

- **Operating System:** Windows
- **Python Version:** 3.8.20
- **Stata Version:** 17
- **Languages:** English, Chinese


### Required Packages
- Python: 3.8.20
- PyTorch: 2.1.2 (CUDA 11.8 + cuDNN 8)
- pandas: 2.0.3
- openpyxl: 3.1.5
- pillow: 10.4.0
- numpy: 1.24.4
- matplotlib: 3.7.2
- scikit-learn: 1.3.0


> 💡 **Recommendation:** Use Anaconda/Miniconda to manage the environment.

---

## 📊 Data Sources

All datasets are publicly available from:
- **Tokenview**
- **Investing.com**
- **Google Trends**
- **Baidu Index**

All data processing pipelines, including cleaning, transformation, and feature engineering, are fully documented in the corresponding code files.

---

## 📁 Repository Structure

Reproducibility_package/<br>
├── Results/<br>
│ ├── Tables/ # Statistical tables from the paper<br>
│ └── Figures/ # All visualizations (JPG format)<br>
│<br>
└── Data&Model/<br>
├── 1 The improved LSTM model/<br>
│ ├── ablation_experiment/ # Ablation study experiments<br>
│ ├── Backtesting system/ # Strategy validation & performance<br>
│ ├── ML/ # Traditional ML baselines<br>
│ ├── train/ # Deep learning training module<br>
│ ├── test/ # Testing & validation scripts<br>
│ └── README-improved LSTM model.md<br>
│<br>
├── 2 Time series model/<br>
│ ├── code/ # Python implementation<br>
│ ├── rawdata/ # Source datasets<br>
│ └── Readme-time series model.txt<br>
│<br>
└── 3 Asset pricing model/<br>
├── code/ # Stata do-files<br>
└── rawdata/ # Source datasets<br>




The Results directory contains the complete set of research outputs:
- All statistical tables referenced in the main text of the paper
- All supplementary tables and figures included in the appendix
- All visualizations and figures in JPG format, corresponding to those presented in the manuscript

## Reproduction Workflow

To regenerate our research results:
1. Execute the code files in each model directory using the corresponding software (Python or Stata)
2. Process the raw data through the analytical pipelines
3. The final outputs will match the tables and figures provided in the Results directory

## The Data Directory

The Data directory houses the research infrastructure organized by methodological approach:

### 1 The improved LSTM model

#### ablation_experiment/

This directory contains the code and experimental results for the Ablation Study. In this project, four different BTC factor feature sets are constructed, each corresponding to different categories of information sources.

The ablation experiments are conducted as follows:
- Train the model using only one group of factors at a time
- Keep the model architecture and training hyperparameters unchanged
- Compare the predictive performance across different factor groups

How to Switch Factor Groups:

To switch between different factor combinations, modify the following file: model/lstm.py

Specifically, locate:

self.lstm = nn.LSTM(input_size=canshu['input_size_1'],)

Parameters to Modify:
- input_size_1
- input_size_2
- input_size_3
- input_size_4

These parameters correspond to the feature dimensions of the four BTC factor groups.

#### Train/

Deep learning training module, including:
- Improved LSTM model definition
- Main training script
- Custom loss functions
- Learning rate scheduling strategy
- Model checkpoint saving (.ckpt)

This module supports further training and reproduction of experiments based on the original dataset.

Dataset Split Configuration:

In config.py, the Group parameter controls different time-based dataset splits:
- Group1-baseline
- Group2-'17-3-3_17-9-3'
- Group3-'17-9-4_18-3-4'
- Group4-'19-7-29_20-1-29'
- Group5-'20-1-30_20-7-30'
- Group6-'21-8-24_22-2-23'
- Group7-'22-2-24_22-8-24'

#### test/

Testing and strategy generation module, including:
- Prediction result validation
- Performance metric calculation
- Confusion matrix visualization
- Trading strategy generation

Reproducing Main Experimental Results:

Run python test.py

Then run: python strategy.py

This script will generate trading signals based on the predicted price movement (up/down).

#### Backtesting system/

Quantitative backtesting module, mainly used for:
- Evaluating trading strategies generated by the model
- Calculating returns and cumulative return curves
- Computing performance metrics such as Maximum Drawdown and Sharpe Ratio
- Visualizing strategy performance

#### ML/

This module reproduces experimental results using traditional machine learning methods.

### 2 Time series model

- code/: Python scripts implementing the time series model
- rawdata/: Source datasets required for training and evaluating the time series model

### 3 Asset pricing model

- code/: Stata do-files implementing the asset pricing model
- rawdata/: Source datasets for data processing, integration, and model estimation

## 📌 Data Usage Notice

> ✅ All datasets necessary to replicate the results in this paper are included within the replication kit.

**Data Accessibility:**
- 🌐 All data is obtained from public platforms
- 🔓 No specific access restrictions

**Important Reminder:**

Users should be mindful of the **terms of use** associated with each platform, which typically govern:
- Commercial use
- Redistribution

**What's Included:**
- Raw data (as retrieved from original sources)
- Processed versions used to generate the paper's results

Both are provided in the replication package for straightforward replication.
