# 3 Asset Pricing Model — Bitcoin Factor Exposure Analysis

## Overview

This directory is part of the reproducibility package for the **"Improved LSTM Model"** paper. It contains the empirical asset pricing analysis of Bitcoin daily returns, implemented entirely in **Stata**.

---

## Directory Structure

```
3 Asset pricing model/
├── code/
│   └── Asset pricing model.do    # Stata master script (~11 KB)
├── rawdata/                        # Raw and intermediate datasets (.dta)
│   ├── 1.dta                       # Merged master dataset (~1.9 MB)
│   ├── btcprice.dta                # Bitcoin daily price series
│   ├── baidu.dta                   # Baidu search volume index
│   ├── google.dta                  # Google Trends search volume
│   ├── sentiment.dta               # Investor sentiment indicators
│   ├── blockchain.dta              # On-chain blockchain metrics
│   ├── cryptocurrency.dta          # Multi-coin market data
│   └── macroeconomics.dta          # Macroeconomic variables
└── result/                         # Outputs
    ├── Decriptive statistics.txt/xls  # Summary statistics (Table 2)
    ├── Figure 3 Price movements of Bitcoin.jpg   # BTC price line chart
    ├── Figure 4 Distribution of daily Bitcoin returns.jpg  # Return histogram
    ├── Table A2.txt/xls            # Traditional factor exposures (CAPM / FF3 / FFC / FF5 / FF6)
    ├── Table A3.txt/xls            # Currency factor exposures (USD / CAD / EUR / GBP / JPY / DKK)
    ├── Table A4.txt/xls            # Investor sentiment (Baidu & Google, lags 1–7 days)
    └── Table A5.txt/xls            # Momentum / autocorrelation (lags 1–7 days)
```

---

## Analysis Pipeline

### Phase 1: Data Processing

Import and clean data from multiple Excel sources:

| Category | Contents |
|----------|----------|
| **Factors** | Fama-French 5-factor + momentum (mkt_rf, smb, hml, rmw, cma, umd) |
| **FX** | USD index + 6 currency pairs (AUD, CAD, EUR, GBP, JPY, DKK) |
| **Commodities** | Brent crude oil, gold, copper futures |
| **Cryptocurrencies** | BTC / ETH / DASH / LTC prices and market caps |

### Phase 2: Data Integration

- Merge all sources by trading date
- Compute log returns: `r_t = ln(P_t / P_{t-1})`
- Construct extreme-return indicators (|r| > 5% / 10% / 20% / 30%)
- Calculate Sharpe ratios and proportion of positive-return days
- Time range: first 853 days trimmed, final sample = **3,318 daily observations**

### Phase 3: Factor Model Regressions

#### Table A2 — Traditional Asset Pricing Factors
Bitcoin's exposure to the Fama-French factor family.  
**Verdict: All insignificant. R² = 0.001–0.002.** BTC returns are essentially unrelated to equity risk factors.

| Model | Factors | R² |
|-------|---------|-----|
| CAPM | mkt_rf | 0.001 |
| FF3 | + smb, hml | 0.002 |
| FFC | + umd | 0.002 |
| FF5 | + rmw, cma | 0.002 |
| FF6 | + umd, rmw, cma | 0.002 |

#### Table A3 — Currency Factors
Bitcoin exposure to 6 major fiat currencies.  
**Verdict: All insignificant. R² = 0.000–0.002.**

#### Table A4 — Investor Sentiment (Table A4)
Baidu Index and Google Trends search volume as predictors, tested at lags 1–7 days.  
**Key finding: Baidu Index at lag 1 is significant at the 1% level (t = -3.21)** with a negative coefficient (β = -6.437). Rising Baidu search interest predicts lower BTC returns the next day — consistent with a "retail FOMO then correction" pattern.

#### Table A5 — Momentum / Autocorrelation
Bitcoin return autocorrelation at lags 1–7 days.  
**Verdict: Completely insignificant.** No evidence of predictable momentum or reversal patterns in daily BTC returns.

---

## Summary of Findings

| Test | Result |
|------|--------|
| Traditional factors (CAPM / FF3 / FF5 / FF6) | ❌ Not significant, R² ≈ 0 |
| FX factors (6 major currencies) | ❌ Not significant |
| Momentum / reversal (lags 1–7) | ❌ Not significant |
| Baidu search sentiment (t+1) | ✅ Significant at 1% level (negative) |
| Google search sentiment | ❌ Largely not significant |

> **Bottom line:** Bitcoin daily returns are almost entirely unexplained by traditional asset pricing factors, consistent with the view that Bitcoin behaves as an independent asset class.

---

## Reproduction Guide

### Requirements
- **Stata** (with `outreg2`, `jb6`, `mergemany` packages installed)
- Original Excel source files must be in the working directory (already converted to `.dta` and stored in `rawdata/`)

### Running the Analysis
Set the working directory to this folder in Stata, then execute:
```stata
do "code/Asset pricing model.do"
```

### Expected Outputs
- All `.txt`, `.xls`, and `.jpg` files under `result/`
- Regression tables and summary statistics printed to the Stata console

---

## Data Dictionary

| File | Description | Size |
|------|-------------|------|
| `1.dta` | Merged master dataset (date / returns / factors / FX / commodities) | 1.9 MB |
| `btcprice.dta` | BTC daily closing prices | 76 KB |
| `baidu.dta` | Baidu search volume index time series | 82 KB |
| `google.dta` | Google Trends time series | 19 KB |
| `blockchain.dta` | On-chain metrics (hash rate, active addresses, etc.) | 380 KB |
| `cryptocurrency.dta` | Multi-coin cross-sectional market data | 165 KB |
| `macroeconomics.dta` | Macroeconomic variables | 464 KB |
| `sentiment.dta` | Composite sentiment indicators | 232 KB |

---

*This directory corresponds to Appendix Tables A2–A5 and Figures 3–4 in the paper.*
