Time Series Forecasting and Trading Strategy Backtesting (BTC Returns)

This project forecasts Bitcoin returns using various time series models and constructs two distinct trading strategies (non-threshold and threshold strategies) based on the predictions. The code covers the complete workflow including data preprocessing, stationarity testing, model training, forecast evaluation, and strategy backtesting.

1. Code Structure
- Data loading and preprocessing
- Return calculation and differencing
- Stationarity and white noise tests
- Rolling forecasts with multiple models
- Forecast accuracy evaluation (RMSE, MAE, MAPE)
- Directional accuracy evaluation
- Two trading strategy simulations:
  - Strategy 1: Simple buy/hold based on predicted sign (non-threshold)
  - Strategy 2: Buy/hold based on 20-day moving average threshold

2. Environment
Ensure the following Python libraries are installed:
```pip install pandas matplotlib statsmodels arch scikit-learn scipy seaborn```

3. Data Description
- Data file: CSV files containing Bitcoin data for different time periods and window lengths
- Field descriptions:
  - `Timestamp`: Time stamp
  - `Weighted_Price`: Weighted price

4. Usage Instructions
(1) Modify the data path to your actual file path and adjust the forecast window length according to the data used:
```data = pd.read_csv('your_data_path.csv', encoding='utf-8')```
```end_loc = corresponding_window_length```
(2) Important: The code contains two identically named `_return` functions (implementing two different strategies). When running, you must manually comment out one strategy's code block; otherwise, the later-defined function will override the earlier one.
(3) Run the main program.
(4) The program will sequentially execute:
   - Data loading and return calculation
   - Stationarity test (ADF)
   - White noise test (Ljung-Box)
   - Normality test (KS test)
   - Rolling forecasts
   - Forecast error evaluation
   - Directional accuracy evaluation
   - Selected strategy backtesting and result saving

5. Model Description
(1) Return Models
- AR models: AR(1), AR(2)
- ARMA models: ARMA(1,1), ARMA(2,2)
- GARCH family models:
  - AR-GARCH(1,1) (1st and 2nd order mean equations)
  - AR-ARCH(1,1) (1st and 2nd order mean equations)
  - AR-EGARCH(1,1) (1st and 2nd order mean equations, evaluation only, not used in strategies)
Note: EGARCH models are excluded from strategy generation due to poor performance (high RMSE and MAPE), but their forecast results are retained for comparative evaluation.
 (2) Forecasting Method
-Rolling window forecasts
-One-step ahead return prediction
(3) Evaluation Metrics
- MSE / RMSE**: Mean squared error and its square root
- MAE: Mean absolute error
- MAPE: Mean absolute percentage error (custom implementation)
- Directional accuracy: Consistency between predicted and actual signs

6. Trading Strategy Details
Strategy 1: Non-threshold Strategy
Trading based on the sign of predicted returns:
- When `predicted return > 0` and no position is held, buy
- When `predicted return < 0` and a position is held, sell
- Otherwise maintain current state (none/keep)

Strategy 2: Threshold Strategy
Threshold based on the average log return of the previous 20 trading days:
- Threshold calculation: `limit = sum(rate[9+n:29+n])`, i.e., the sum of returns over the previous 20 days from the current time point
- When `predicted return > 0` and `predicted return > threshold` and no position is held, buy
- When `predicted return < 0` and a position is held, sell
- Otherwise maintain current state (none/keep)

7. Important Notes
(1) Strategy selection: The code contains two identically named `_return` functions. You must manually comment out one of them during runtime; otherwise, the later-defined function will override the earlier one.
(2) EGARCH exclusion: Although EGARCH models are estimated and evaluated, they are not used in strategy generation due to large forecast errors.