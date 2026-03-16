# backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import config as C
import utils as u
import os
import random
import logging

# ===============================
# Set Random Seed
# ===============================
random.seed(C.random_seed)
np.random.seed(C.random_seed)

# ===============================
# Logging Configuration
# ===============================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# =====================================================
# Utility Functions
# =====================================================

def len_date(strategy):
    """
    Get the maximum and minimum date from a strategy.
    """
    if not strategy:
        raise ValueError("Strategy is empty, cannot determine date range.")

    date_list = [tt[0] for buy_sell in strategy.values() for tt in buy_sell]
    return max(date_list), min(date_list)


def strategy_change(strategy):
    """
    Sort strategy dictionary by date.
    """
    if not strategy:
        print("Strategy is empty.")
        return {}

    return dict(sorted(strategy.items()))


# =====================================================
# Backtesting Engine (With Transaction Cost)
# =====================================================

def trade_loop_back1(strategy, date_array,
                     initial_money=10000,
                     fee_rate=0.01):
    """
    Backtesting engine (All-In strategy) with transaction cost.
    """

    money = initial_money
    stock_holdings = {}
    asset = {}
    daily_return = {}
    cumulative_yield = {}
    total_fees = 0.0
    total_return = 1

    # Initialize holdings
    for date in date_array:
        for action in strategy.get(date, []):
            stock = action[2]
            if stock not in stock_holdings:
                stock_holdings[stock] = 0

    for idx, date in enumerate(date_array):

        asset[date] = money
        daily_return[date] = 0

        for action in strategy.get(date, []):
            act = action[0].lower()
            stock = action[2]

            current_price = u.data_acquisition(date, stock)
            if current_price == -1:
                logging.warning(f"Missing price for {stock} on {date}")
                continue

            # ================= BUY =================
            if act == "buy":

                denominator = current_price * (1 + fee_rate)
                if denominator == 0:
                    continue

                quantity = money // denominator

                if quantity > 0:
                    total_cost = current_price * quantity
                    fee = total_cost * fee_rate
                    total_cost_with_fee = total_cost + fee

                    if money >= total_cost_with_fee:
                        money -= total_cost_with_fee
                        stock_holdings[stock] += quantity
                        total_fees += fee

            # ================= SELL =================
            elif act == "sell":

                quantity = stock_holdings.get(stock, 0)
                if quantity > 0:
                    revenue = current_price * quantity
                    fee = revenue * fee_rate
                    money += revenue - fee
                    stock_holdings[stock] = 0
                    total_fees += fee

        # ================= Asset Calculation =================
        total_asset = money
        for s, qty in stock_holdings.items():
            if qty > 0:
                price = u.data_acquisition(date, s)
                if price != -1:
                    total_asset += price * qty

        asset[date] = total_asset

        if idx > 0:
            previous_asset = asset[date_array[idx - 1]]
            daily_return[date] = (total_asset - previous_asset) / previous_asset
            total_return *= (1 + daily_return[date])
            cumulative_yield[date] = total_return - 1
        else:
            cumulative_yield[date] = 0

    return calculate_metrics(cumulative_yield, daily_return, money)


# =====================================================
# Backtesting Engine (Without Transaction Cost)
# =====================================================

def trade_loop_back2(strategy, date_array,
                     initial_money=10000):
    """
    Backtesting engine (All-In strategy) without transaction cost.
    """

    money = initial_money
    stock_holdings = {}
    asset = {}
    daily_return = {}
    cumulative_yield = {}
    total_return = 1

    for date in date_array:
        for action in strategy.get(date, []):
            stock = action[2]
            if stock not in stock_holdings:
                stock_holdings[stock] = 0

    for idx, date in enumerate(date_array):

        asset[date] = money
        daily_return[date] = 0

        for action in strategy.get(date, []):
            act = action[0].lower()
            stock = action[2]

            current_price = u.data_acquisition(date, stock)
            if current_price == -1:
                continue

            if act == "buy":
                quantity = money // current_price
                if quantity > 0:
                    money -= current_price * quantity
                    stock_holdings[stock] += quantity

            elif act == "sell":
                quantity = stock_holdings.get(stock, 0)
                if quantity > 0:
                    money += current_price * quantity
                    stock_holdings[stock] = 0

        total_asset = money
        for s, qty in stock_holdings.items():
            if qty > 0:
                price = u.data_acquisition(date, s)
                if price != -1:
                    total_asset += price * qty

        asset[date] = total_asset

        if idx > 0:
            previous_asset = asset[date_array[idx - 1]]
            daily_return[date] = (total_asset - previous_asset) / previous_asset
            total_return *= (1 + daily_return[date])
            cumulative_yield[date] = total_return - 1
        else:
            cumulative_yield[date] = 0

    return calculate_metrics(cumulative_yield, daily_return, money)


# =====================================================
# Performance Metrics Calculation
# =====================================================

def calculate_metrics(cumulative_yield, daily_return, money):

    returns_series = pd.Series(daily_return)
    yield_series = pd.Series(cumulative_yield)

    rolling_max = yield_series.cummax()
    drawdown = rolling_max - yield_series
    max_drawdown = drawdown.max()

    num_days = len(yield_series)
    total_return = yield_series.iloc[-1] + 1

    annualized_return = (
        math.pow(total_return, 360 / num_days) - 1
        if num_days > 0 else 0
    )

    risk_free_rate = 0.0176
    returns_std = returns_series.std()

    sharpe_ratio = (
        (annualized_return - risk_free_rate)
        / (returns_std * math.sqrt(360))
        if returns_std != 0 else np.nan
    )

    er_ratio = (
        annualized_return / max_drawdown
        if max_drawdown != 0 else np.nan
    )

    var1 = np.percentile(returns_series.dropna(), 1)
    var5 = np.percentile(returns_series.dropna(), 5)

    return yield_series, sharpe_ratio, max_drawdown, \
           annualized_return, money, er_ratio, var1, var5


# =====================================================
# Main Function
# =====================================================

def main():

    os.makedirs(C.result_plot_dir, exist_ok=True)

    strategy_names = [
        'BTC_the_up_down_Group1_threshold_0.5',
        'BTC_the_up_down_Group2_threshold_0.5',
        'BTC_the_up_down_Group3_threshold_0.5',
        'BTC_the_up_down_Group4_threshold_0.5',
        'BTC_the_up_down_Group5_threshold_0.5',
        'BTC_the_up_down_Group6_threshold_0.5',
        'BTC_the_up_down_Group7_threshold_0.5'
    ]

    results = []

    for strategy_name in strategy_names:

        strategy_file = os.path.join(
            C.strategy_dir,
            f"{strategy_name}.txt"
        )

        try:

            strategy = u.load_strategy(strategy_file)
            if not strategy:
                continue

            strategy_sorted = strategy_change(strategy)

            data_handler = u.DataHandle(strategy_sorted)
            date_array = data_handler.date_array

            yield_series, sharpe_ratio, max_drawdown, \
            annualized_return, final_money, er_ratio, \
            var1, var5 = trade_loop_back2(
                strategy_sorted,
                date_array,
                initial_money=10000000
            )
            #0.001fee_rate
            # yield_series, sharpe_ratio, max_drawdown, \
            # annualized_return, final_money, er_ratio, \
            # var1, var5=trade_loop_back1(strategy, date_array, initial_money=10000000, fee_rate=0.001)
            # u.plot_profit(
                # yield_series,
                # strategy_name,
                # C.result_plot_dir
            # )

            result = {
                "Strategy": strategy_name,
                "Cumulative Return": f"{yield_series.iloc[-1]:.2%}",
                "Sharpe Ratio": f"{sharpe_ratio:.4f}",
                "Max Drawdown": f"{max_drawdown:.2%}",
                "Annualized Return": f"{annualized_return:.2%}",
                "Final Capital": f"{final_money:.2f}",
                "Return/Risk": f"{er_ratio:.4f}",
                "VaR 1%": f"{var1:.4f}",
                "VaR 5%": f"{var5:.4f}"
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing {strategy_name}: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(
        C.result_csv,
        "backtest_results.csv"
    )
    results_df.to_csv(csv_path,
                      index=False,
                      encoding='utf-8-sig')

    print("Backtest completed successfully.")


if __name__ == "__main__":
    main()