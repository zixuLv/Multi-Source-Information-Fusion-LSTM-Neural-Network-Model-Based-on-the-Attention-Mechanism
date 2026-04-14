
I. Development Environment

This project was developed and tested under the following environment:

Operating System: Windows

Python Version: 3.8.20

II. Core Dependencies

Python==3.8.20
PyTorch==2.1.2（CUDA 11.8 + cuDNN 8）
pandas==2.0.3
openpyxl==3.1.5
pillow==10.4.0
numpy==1.24.4
matplotlib==3.7.2
scikit-learn==1.3.0

III. Project Structure

project_root/
│
├── ablation_experiment/      # Ablation study code
│
├── Backtesting system/       # Backtesting module (strategy validation & performance evaluation)
│
├── ML/                       # Traditional machine learning models
│
├── train/                    # Deep learning training module (training scripts)
│
├── test/                     # Testing scripts and independent experiment validation
│
└── README.md                 # Project documentation

IV. Module Description
## Train/

This module implements the complete deep learning training pipeline and supports full reproduction of experiments based on the provided dataset.

---

### Contents

- **Improved LSTM model definition**: `model/lstm.py`  
- **Main training script**: `main.py`  
- **Utility functions**: `utils.py` (shared across the project)  
- **Global configuration file**: `config.py` (for parameter settings and data path configuration)  
- **Model checkpoint saving**: `.ckpt` files are automatically generated during training  

---

### Data Source

The dataset used in this project is included in the repository. It consists of multi-source Bitcoin (BTC) factor features derived from different categories of information. The dataset is located at:

```
train/dataset/dataset
```

Specifically:

- `train/data/suoyin.xls` serves as the index file, used to organize and locate different feature data  

The data can be directly used for model training and evaluation.

Unless otherwise specified, **no additional data download is required**.

---

### Usage

Run the following command to start training:

```bash
python main.py
```

The program will automatically perform the following steps:

- Data loading  
- Model initialization  
- Model training and validation  
- Saving model checkpoints and results  

---

### Output Description

- Key training results can be viewed in the **command-line output**  
- Model checkpoint files (`.ckpt`) will be automatically saved in:

```
train/ckpt
```

---

### Dataset Split Configuration

The dataset is split based on time periods, controlled by the `Group` parameter in `config.py`:

- **Group1** — baseline  
- **Group2** — '17-3-3_17-9-3'  
- **Group3** — '17-9-4_18-3-4'  
- **Group4** — '19-7-29_20-1-29'  
- **Group5** — '20-1-30_20-7-30'  
- **Group6** — '21-8-24_22-2-23'  
- **Group7** — '22-2-24_22-8-24'  

## test/

This module is used for model evaluation, result analysis, and trading strategy generation. It supports the reproduction of the main experimental results reported in the paper.

---

### Contents

- **Prediction validation**: Evaluate model prediction results  
- **Performance metrics computation**: Calculate key metrics such as accuracy  
- **Confusion matrix visualization**: Visualize classification performance  
- **Trading strategy generation**: Generate trading signals based on predictions  

---

### Data Description

The testing data shares the same source as the training data, consisting of multi-source Bitcoin (BTC) factor features.  
The data loading and preprocessing pipeline is consistent with the training stage and is controlled by `config.py`.

---

### Usage

To reproduce the main experimental results, run the scripts in the following order:

---

#### Step 1: Model Testing

```bash
python test.py
```
The program will automatically:

Iterate over all Group configurations (time-based splits)
Split the dataset according to each Group
Load the corresponding model or perform training-based evaluation
Compute prediction results and performance metrics
Save evaluation results and visualizations

Output Structure

Each Group will generate a separate result directory:
test_results/
 ├── Group1/
 │   ├── test_results.csv        # Predictions (true labels vs predicted labels)
 │   ├── test_metrics.csv        # Performance metrics (e.g., accuracy)
 │   ├── confusion_matrix.png    # Confusion matrix visualization
 │
 ├── Group2/
 │   └── ...
Explanation:

test_metrics.csv: stores key evaluation metrics（Main experimental results）
confusion_matrix.png: visualization of classification results（（Main experimental results））
test_results.csv: detailed prediction results and corresponding thresholds

#### Step 2: Strategy Generation
```bash
python strategy.py
```
The program will automatically:

Iterate over test results for all Groups
Generate trading signals based on test_results.csv
Save the corresponding trading strategy outputs

Strategy Output Example:
test/output_strategy/
 └── BTC_the_up_down_Group1_threshold_0.5.txt
 
Parameter Description (Important)
threshold: controls the decision boundary for predicting up/down movements (default: 0.5)
This parameter can be modified in the main() function of strategy.py to generate different trading strategies



## Backtesting system/

This module implements a quantitative backtesting framework for evaluating trading strategies generated by the model.

---

### Functions

- **Strategy evaluation**: Evaluate trading strategies generated by the model  
- **Return calculation**: Compute returns and cumulative return curves  
- **Performance metrics**: Calculate key financial indicators, including:  
  - Maximum Drawdown  
  - Sharpe Ratio  
- **Visualization**: Generate performance plots for strategy analysis  

---

### Input Data Description

#### 1. Strategy Data

Generated trading strategies should be placed in the following directory:

Backtesting system/output_strategy/

Example file:
BTC_the_up_down_Group1_threshold_0.5.txt

#### 2. Market Price Data

Historical BTC price data is located at:
Backtesting system/data/btc.xls

### Usage

Run the following command to perform backtesting:

```bash
python backtest.py
```
#### Backtesting Modes
Default Mode (No Transaction Cost)
The script runs backtesting without transaction fees by default.
With Transaction Cost
To include transaction fees:
Enable trade_loop_back1()
Comment out trade_loop_back2()
(Typically around line ~301 in backtest.py)

Output：
1. Backtesting Results (Main Experimental Results)
results/backtest_results.csv

2. Return Curve Visualization (Main Experimental Results)
results/BTC_the_up_down_Group1_threshold_0.5.png



##ablation_experiment/
This directory contains the code and experimental results for the Ablation Study.

In this project, four different BTC factor feature sets are constructed, each corresponding to different categories of information sources.

The ablation experiments are conducted as follows:

Train the model using only one group of factors at a time

Keep the model architecture and training hyperparameters unchanged

Compare the predictive performance across different factor groups

How to Switch Factor Groups

To switch between different factor combinations, modify the following file:
model/lstm.py
Specifically, locate:
self.lstm = nn.LSTM(
    input_size=canshu['input_size_1'],
)
Parameters to Modify

input_size_1
input_size_2
input_size_3
input_size_4
These parameters correspond to the feature dimensions of the four BTC factor groups.

### Usage

```bash
python main.py
```
Output：

- **Accuracy** is obtained directly from the command-line output  

- **Confusion matrix** is saved at: 
ablation experiment/baseline/best_confusion_matrix 

- **Detailed prediction results (up/down)** are stored in: 
ablation experiment/baseline/best_test_results.xlsx 

## ML/

This module reproduces the experimental results using traditional machine learning methods.

---

### Usage

Navigate to the following directory:
final_code/ML/ml/

Run the main script:

```bash
python main.py
```
Output
Accuracy statistics are saved in:
final_code/ML/ml/TEST_result.txt

#Recommendation: To reproduce the experimental results in this paper, please follow the steps below:

Run python test.py in the test directory to obtain model prediction results and related evaluation metrics;
Run python strategy.py to enter the strategy generation stage, where two types of strategies (default threshold and 0.6 threshold) are generated for each of the seven time groups;
In the backtesting stage, it is recommended to first perform backtesting without transaction costs, and then enable the transaction cost mode for comparative analysis;

The resulting return curves and performance metrics constitute the core experimental results of this study.