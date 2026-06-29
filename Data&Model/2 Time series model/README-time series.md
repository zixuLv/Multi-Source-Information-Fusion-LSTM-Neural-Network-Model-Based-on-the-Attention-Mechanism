# BTC Time Series Prediction Model

Bitcoin price direction forecasting using ARIMA/ARCH/GARCH family models,
with threshold / no-threshold buy-and-hold trading strategies and backtesting.

Author: zyh (2022-08-07)
Original conda environment: 4.13.0

---

## 1. Project Overview

### 1.1 Models (10 total)

| # | Model | Family |
|---|-------|--------|
| 1 | AR(1) | Autoregressive |
| 2 | AR(2) | Autoregressive |
| 3 | ARMA(1,1) | Autoregressive Moving Average |
| 4 | ARMA(2,2) | Autoregressive Moving Average |
| 5 | AR(1)-ARCH | AR + Conditional Heteroskedasticity |
| 6 | AR(2)-ARCH | AR + Conditional Heteroskedasticity |
| 7 | AR(1)-GARCH | AR + Generalized ARCH |
| 8 | AR(2)-GARCH | AR + Generalized ARCH |
| 9 | AR(1)-EGARCH | AR + Exponential GARCH |
| 10 | AR(2)-EGARCH | AR + Exponential GARCH |

EGARCH models were excluded from strategy backtesting due to poor directional accuracy.

### 1.2 Time Periods (5 sample periods)

The paper defines five sample periods: baseline, before/after Event 1, before/after Event 2.

| Period | Start | End | Context |
|--------|-------|-----|---------|
| **Baseline** | 2024-04-01 | 2024-09-30 | Reference period |
| **Event 1 — before** | 2019-07-29 | 2020-01-29 | Pre-COVID-19 |
| **Event 1 — after** | 2020-01-30 | 2020-07-30 | Post-COVID-19 outbreak |
| **Event 2 — before** | 2021-08-24 | 2022-02-23 | Pre-Fed rate hikes |
| **Event 2 — after** | 2022-02-24 | 2022-08-24 | Post-Fed rate hikes |

### 1.3 Rolling Windows (4 sizes)

18, 30, 60, 90 trading days.

### 1.4 Data Preprocessing

For each rolling window, an extra "window size + 1 day" of data is prepended to the period
for model initialization. Example:

- Baseline period: **2024-04-01 ~ 2024-09-30**
- `baseline-30.csv` contains data from **2024-03-01 ~ 2024-09-30** (30-day window + 1 lead day)

All window CSVs were pre-split according to this rule and stored in `data/windows/`.

---

## 2. Directory Structure

```
BTC_TS_Model/
|
|-- README.md                             # This file (final submission version)
|-- table_file_mapping.txt                # Table-to-file cross-reference
|
|-- code/                                 # Source code
|   |-- 01_prediction_accuracy.py         # Step 1: prediction accuracy evaluation
|   |-- 02_strategy_no_threshold.py       # Step 2a: no-threshold strategy
|   |-- 02_strategy_threshold.py          # Step 2b: threshold strategy
|   |-- backtesting/
|       |-- backtest.py                   # Step 3: backtesting engine
|       |-- config.py                     # Backtest config (do not run directly)
|       |-- utils.py                      # Backtest utilities (do not run directly)
|
|-- data/                                 # Input data
|   |-- raw/
|   |   |-- btc.csv                       # BTC full price history
|   |   |-- btc2015-08-10..2021-12-17.csv # BTC subset (2015-2021)
|   |   |-- suoyin.csv                    # Index / benchmark data
|   |-- windows/                          # Pre-split time-window data (20 files)
|       |-- baseline-{18,30,60,90}.csv
|       |-- btc-beforeevent1-{18,30,60,90}.csv
|       |-- btc-afterevent1-{18,30,60,90}.csv
|       |-- btc-beforeevent2-{18,30,60,90}.csv
|       |-- btc-afterevent2-{18,30,60,90}.csv
|
|-- outputs/                              # Generated outputs
    |-- accuracy/
    |   |-- MAPE-RMSE-Accuracy-Results.docx # Manually aggregated accuracy records
    |-- strategies/
    |   |-- no_threshold/                  # No-threshold strategy signals (40 .txt)
    |   |-- threshold/                     # Threshold strategy signals (40 .txt)
    |-- backtest/
    |   |-- backtest_results.csv           # Backtest results (no transaction fee)
    |   |-- backtest_results_fee.csv       # Backtest results (with transaction fee)
    |-- figures/
        |-- no_threshold/                  # No-threshold forecast charts (40 .png)
        |-- threshold/                     # Threshold forecast charts (40 .png)
```

---

## 3. Workflow — Three-Step Pipeline

### Step 1: Prediction Accuracy Evaluation

**Script**: `code/01_prediction_accuracy.py`

Runs all 10 models across 5 sample periods x 4 rolling windows = 20 combinations.
Evaluation metrics: MSE, RMSE, MAE, MAPE, directional (up/down) prediction accuracy.

**Before each run, modify two lines in the script**:
- **Line 35**: data path -> `data/windows/{period}-{window}.csv`
- **Line 91**: `end_loc` -> window size (18, 30, 60, or 90)

```
python code/01_prediction_accuracy.py
```

> Results are printed directly to console (stdout).
> After each run, **manually record** the output into
> `outputs/accuracy/MAPE-RMSE-Accuracy-Results.docx`.

**Paper table mapping**:

| Runs | Paper Table |
|------|:----------:|
| baseline-{18,30,60,90} | **Table 3** — Accuracy across rolling windows |
| baseline-18 | **Table 4** — RMSE & MAPE |
| baseline-18 | **Table 6** — Best models, baseline accuracy |
| {before,after}event{1,2}-18 | **Table 7** — Event-period accuracy |

> Full file-to-table mapping: `table_file_mapping.txt`.


### Step 2: Strategy Signal Generation

**Scripts**: `code/02_strategy_no_threshold.py` / `code/02_strategy_threshold.py`

The **18-day rolling window** performed best across all models, and **EGARCH models
were excluded** due to poor performance. Strategy scripts therefore only use 18-day
window data for the 8 non-EGARCH models across 5 periods.

**Before each run, modify**:
- **Line 35**: data path -> `data/windows/{period}-18.csv`
- **Line 91**: `end_loc = 18` (fixed after first set)

```
python code/02_strategy_no_threshold.py    # Without threshold
python code/02_strategy_threshold.py       # With threshold
```

> **Overwrite warning**: Each run **overwrites** the previous output file.
> After each execution, immediately **rename** the output file with the period suffix
> (baseline / beforeevent1 / afterevent1 / beforeevent2 / afterevent2).

**Strategy models** (8 of 10):
AR(1), AR(2), ARMA(1,1), ARMA(2,2), AR(1)-ARCH, AR(2)-ARCH, AR(1)-GARCH, AR(2)-GARCH

Each model produces one strategy signal file (`.txt`) with format
`[date, buy/sell/keep/none, position_status]`.
Total: 80 strategy files = 8 models x 5 periods x 2 strategy types.


### Step 3: Backtesting

**Script**: `code/backtesting/backtest.py`
**Dependencies**: `code/backtesting/config.py` + `code/backtesting/utils.py`
(do not run them directly; they must be present in the same directory.)

**Prerequisite**: Copy all strategy files (`.txt`) from Step 2 into
`code/backtesting/strategies/`.

**Must be run from the `code/backtesting/` directory.**

**Two backtesting functions**:

| Function | Transaction Fee | Line Pair | Output File |
|----------|:---:|-----------|-------------|
| `trade_loop_back1` | With fee | line 371 + line 403 | `backtest_results_fee.csv` |
| `trade_loop_back2` | No fee | line 372 + line 404 | `backtest_results.csv` |

Comment/uncomment the appropriate function calls (lines 371-372 and 403-404),
then run each separately.

```
cd code/backtesting
python backtest.py
```

**Paper table mapping**:

| Output File | Paper Tables |
|-------------|:------------:|
| `backtest_results.csv` | **Table 8**, Table A7, Table A14 Panel A |
| `backtest_results_fee.csv` | Table A14 Panel B, Table A13 |


---

## 4. Dependencies

**Exact reproduction environment (conda 4.13.0):**

| Package | Version |
|---------|---------|
| Python | 3.9.7 |
| pandas | 1.4.3 |
| numpy | 1.21.5 |
| statsmodels | 0.13.2 |
| arch | 5.3.1 |
| scipy | 1.7.3 |
| scikit-learn | 1.1.1 |
| matplotlib | 3.5.1 |
| seaborn | 0.11.2 |

```bash
conda create -n btc_ts python=3.9.7
conda activate btc_ts
conda install pandas=1.4.3 numpy=1.21.5 statsmodels=0.13.2 scipy=1.7.3 scikit-learn=1.1.1 matplotlib=3.5.1 seaborn=0.11.2
conda install -c conda-forge arch=5.3.1
```

> **Note:** Results are sensitive to the `arch` package version. The forecast extraction code (`temp.iloc[i+end_loc-1]`) depends on internal API structure that changed between arch 5.x and later versions. Reproduction must use arch 5.3.1.

---

## 5. Important Notes

1. **Hardcoded paths**: The original scripts use `D:/` paths. Update to match your
   environment before running.
2. **Console-only output**: Prediction accuracy results are printed to stdout only;
   manually record them after each run.
3. **Strategy overwrite trap**: Each strategy run overwrites the previous output.
   Always rename the output file with the period suffix immediately after each run.
4. **EGARCH excluded**: EGARCH models were removed from strategies and backtesting
   due to catastrophic RMSE (155-15,288) and near-random directional accuracy.
5. **18-day window optimal**: 18-day rolling window achieved the best prediction
   accuracy; strategies and backtesting use 18-day data exclusively.
6. **pandas 2.x compatibility**: The original code (written 2022) requires two
   patches for pandas >= 2.0 (see `E:/桌面/验证论文/code/generate_tables.py`).

---

## 6. Paper Table Quick Reference

| Table | Content | Data Source |
|:-----:|------|------|
| Table 3 | 10 models x 4 windows accuracy | Step 1: baseline-{18,30,60,90} |
| Table 4 | RMSE & MAPE (18-day, baseline) | Step 1: baseline-18 |
| Table 6 | Best models up/down/overall accuracy | Step 1: baseline-18 |
| Table 7 | Event-period up/down accuracy | Step 1: event{1,2}-18 |
| Table 8 | Backtest performance (no fee) | Step 2+3: backtest_results.csv |

See `table_file_mapping.txt` for detailed file-path correspondence.
