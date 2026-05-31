# utils.py

import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import config as C
import random
import ast
import os

# Global cache for BTC data to avoid repeated file loading
_btc_data_cache = None


def date2num(date_str):
    """
    Convert date string to integer format yyyymmdd.
    """
    try:
        return int(pd.to_datetime(date_str).strftime('%Y%m%d'))
    except ValueError:
        raise ValueError("Invalid date format. Please use 'YYYY-MM-DD' or similar format.")


def num2date(date_num):
    """
    Convert integer format yyyymmdd to date string 'YYYY-MM-DD'.
    """
    year = date_num // 10000
    month = (date_num % 10000) // 100
    day = date_num % 100
    return f"{year}-{month:02d}-{day:02d}"


def load_strategy(file_path):
    """
    Load strategy from a text file.

    Example file format:
    btc     [[20200203, 'buy', 1], [20200204, 'keep', 0], ..., [20200712, 'none', 0]]
    """
    strategy = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Assume asset name and action list are separated by whitespace
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

                stock, actions_str = parts
                try:
                    # Safely parse action list
                    actions = ast.literal_eval(actions_str)
                    if not isinstance(actions, list):
                        print(f"Invalid action list format: {actions_str}")
                        continue

                    for action in actions:
                        if not (isinstance(action, list) and len(action) == 3):
                            print(f"Skipping malformed action: {action}")
                            continue

                        date, act, qty = action

                        if not isinstance(date, int):
                            print(f"Skipping action with invalid date format: {action}")
                            continue

                        if date not in strategy:
                            strategy[date] = []

                        strategy[date].append([act, qty, stock])

                except (SyntaxError, ValueError) as e:
                    print(f"Failed to parse action list: {e}, content: {actions_str}")
                    continue

    except FileNotFoundError:
        print(f"Strategy file not found: {file_path}")
    except Exception as e:
        print(f"Failed to load strategy file: {e}")

    return strategy


def data_acquisition(time, data_name):
    """
    Retrieve price data for a specific date and asset from CSV file.
    """
    global _btc_data_cache

    if _btc_data_cache is None:
        try:
            _btc_data_cache = pd.read_csv(C.data_paths["btc_data"])
            print(f"BTC data loaded successfully, total records: {len(_btc_data_cache)}")

            # Create 'Date_Num' column if not exists
            if 'Date_Num' not in _btc_data_cache.columns:
                if 'Timestamp' in _btc_data_cache.columns:
                    _btc_data_cache['Date_Num'] = _btc_data_cache['Timestamp'].apply(lambda x: date2num(x))
                    print("'Date_Num' column created from 'Timestamp'.")
                else:
                    print("BTC data file is missing 'Date_Num' and 'Timestamp' columns.")
                    return -1

            if 'Weighted_Price' not in _btc_data_cache.columns:
                print("BTC data file is missing 'Weighted_Price' column.")
                return -1

        except FileNotFoundError:
            print(f"BTC data file not found: {C.data_paths['btc_data']}")
            return -1
        except Exception as e:
            print(f"Failed to load BTC data: {e}")
            return -1

    # Filter data by date
    filtered_data = _btc_data_cache[_btc_data_cache['Date_Num'] == time]

    if filtered_data.empty:
        print(f"Date mismatch:\n\n{data_name}: {time} does not exist")
        return -1
    else:
        price = filtered_data['Weighted_Price'].values[0]
        print(f"Retrieved price of {data_name} on {time}: {price}")
        return price


def dict2csv(dic, filename):
    """
    Write dictionary to CSV file.
    All dictionary value lists must have equal length.
    """
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
            csv_writer.writeheader()
            for i in range(len(dic[list(dic.keys())[0]])):
                row = {key: dic[key][i] for key in dic.keys()}
                csv_writer.writerow(row)
    except Exception as e:
        print(f"Failed to write CSV file: {e}")


def plot_profit(yield_series, strategy_name, save_dir):
    """
    Plot cumulative profit curve and save as PNG file.
    """
    try:
        dates = [num2date(day) for day in sorted(yield_series.keys())]
        profits = [yield_series[day] for day in sorted(yield_series.keys())]

        plt.rcParams.update({'font.size': 16})

        plt.figure(figsize=(12, 9))
        plt.plot(profits, color='purple', label='Strategy', linewidth=2.5)

        plt.xlabel("Date", fontsize=20)
        plt.ylabel("Accumulated Profit", fontsize=20)
        plt.title(f"Cumulative Profit: {strategy_name}", fontsize=18)

        if len(profits) > 10:
            ticks = np.linspace(0, len(profits) - 1, 10).astype(int)
        else:
            ticks = range(len(profits))

        plt.xticks(ticks=ticks,
                   labels=[dates[i] for i in ticks],
                   rotation=45,
                   fontsize=20)

        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{strategy_name}.png")
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"Plotting failed: {e}")


class DataHandle:
    """
    Handle strategy data and initialize date, stock,
    action, and quantity arrays.
    """

    def __init__(self, strategy):
        self.strategy = strategy
        self.date_array = self._init_days()
        self.stock_array = self._init_stock()
        self.action_array = self._init_action()
        self.number_array = self._init_number()

    def _init_days(self):
        return sorted(self.strategy.keys())

    def _init_stock(self):
        stock_array = {}
        for date in self.strategy:
            stock_array[date] = [action[2] for action in self.strategy[date]]
        return stock_array

    def _init_action(self):
        action_array = {}
        for date in self.strategy:
            action_array[date] = [action[0] for action in self.strategy[date]]
        return action_array

    def _init_number(self):
        number_array = {}
        for date in self.strategy:
            number_array[date] = [action[1] for action in self.strategy[date]]
        return number_array