import config as C
import pandas as pd
import glob
import os
import torch
import numpy as np
import utils as U
import logging


def save_strategy(strategy, i, output_dir="output_strategy"):
    """
    Save the generated trading strategy to a text file.

    Args:
        strategy (dict): Dictionary containing trading actions.
        i (str): Identifier (e.g., group name and threshold).
        output_dir (str): Directory where the strategy file will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, f"BTC_the_up_down_{i}.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        for stock, actions in strategy.items():
            f.write(f"{stock}\t{actions}\n")

    logging.info(f"Strategy saved to: {file_path}")


def date_to_num(date: str) -> int:
    """
    Convert a date string in format 'YYYY/MM/DD' to an integer format YYYYMMDD.

    Args:
        date (str): Date string in format 'YYYY/MM/DD'.

    Returns:
        int: Integer representation of the date.

    Raises:
        ValueError: If the date format is invalid.
    """
    try:
        year, month, day = map(int, date.split("/"))
    except ValueError:
        raise ValueError("Invalid date format. Example: 2015/09/10")
    return year * 10000 + month * 100 + day


def process_strategy(f_p, zz, base_money, fdd):
    """
    Execute backtesting based on predicted trading signals.

    Args:
        f_p (list): Price series.
        zz (list): Trading signals (1 = buy, 0 = sell).
        base_money (float): Initial capital.
        fdd (list): Date series.

    Returns:
        tuple: (strategy dictionary, up/down record dictionary)
    """
    strategy = {'btc': []}
    up_down = {'lstm': []}
    cash = base_money      # Current cash balance
    holdings = 0           # Current asset holdings

    for i in range(len(f_p)):
        s = []
        p = []

        # Ensure price is numeric
        try:
            price = float(f_p[i])
        except ValueError:
            logging.error(f"Invalid price data: {f_p[i]}, skipping.")
            continue

        # Buy signal
        if zz[i] == 1:
            if cash > 0:
                buy_num = cash / price   # All-in buying
                holdings += buy_num
                cash = 0

                s.append(date_to_num(fdd[i]))
                s.append('buy')
                s.append(1)

                p.append(date_to_num(fdd[i]))
                p.append(1)

                up_down['lstm'].append(p)
                strategy['btc'].append(s)
            else:
                strategy['btc'].append(
                    [date_to_num(fdd[i]), 'keep', 'none']
                )

        # Sell signal
        elif zz[i] == 0:
            if holdings > 0:
                sell_num = holdings
                cash += sell_num * price
                holdings = 0

                s.append(date_to_num(fdd[i]))
                s.append('sell')
                s.append(0)

                p.append(date_to_num(fdd[i]))
                p.append(0)

                up_down['lstm'].append(p)
                strategy['btc'].append(s)
            else:
                strategy['btc'].append(
                    [date_to_num(fdd[i]), 'keep', 'none']
                )

    # Force liquidation at the final time step
    if holdings > 0:
        sell_num = holdings
        final_price = float(f_p[-1])
        cash += sell_num * final_price
        holdings = 0

        final_date = date_to_num(fdd[-1])
        strategy['btc'].append([final_date, 'sell', 0])
        up_down['lstm'].append([final_date, 0])

    return strategy, up_down


def label_tal(dier: str):
    """
    Load label and date data from CSV file.

    Args:
        dier (str): File path.

    Returns:
        tuple: (label list, date list)
    """
    suoyin_data = pd.read_csv(dier)
    label_list = list(map(float, suoyin_data.values[0]))
    date_list = list(suoyin_data.values[1])
    return label_list, date_list


def generate_labels(zz, threshold):
    """
    Generate binary trading signals based on probability threshold.

    Args:
        zz (list): List of (prob_down, prob_up).
        threshold (float): Decision threshold.

    Returns:
        list: Binary signals (0 = sell, 1 = buy).
    """
    return [0 if z[0] > threshold else 1 for z in zz]


def load_test_results(file_path: str) -> pd.DataFrame:
    """
    Load model test results from CSV file.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: DataFrame containing prediction results.

    Raises:
        IOError: If loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = {
            'True Label',
            'Predicted Label',
            'Probability Class 0',
            'Probability Class 1'
        }
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        return df
    except Exception as e:
        raise IOError(f"Failed to read file: {e}")


def load_price_data(file_path: str) -> pd.DataFrame:
    """
    Load price data file and return a DataFrame with dates and prices.

    The CSV file is assumed to have:
    - First row: dates
    - Second row: prices

    Args:
        file_path (str): Path to price data file.

    Returns:
        pd.DataFrame: DataFrame with 'date' and 'price' columns.
    """
    try:
        df = pd.read_csv(file_path, header=None)

        if df.shape[0] < 2:
            raise ValueError(f"Invalid format in {file_path}, insufficient data.")

        dates = df.iloc[0].tolist()
        prices = df.iloc[1].tolist()

        price_df = pd.DataFrame({
            'date': dates,
            'price': prices
        })

        return price_df

    except Exception as e:
        logging.error(f"Failed to load price data: {e}")
        raise


def main():
    """
    Batch backtesting across multiple groups defined in config.py.
    """

    base_money = 10000000.0
    threshold = 0.5

    price_data_file = r"data\suoyin.csv"
    test_root = r"test_results"
    output_dir = "output_strategy"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load price data once
    try:
        price_df = load_price_data(price_data_file)
    except IOError as e:
        logging.error(e)
        return

    dates = price_df['date'].tolist()
    prices = price_df['price'].tolist()

    # Iterate through all configured groups
    for group_name, group_cfg in C.GROUPS.items():

        print(f"\nProcessing {group_name}")

        x = group_cfg.x + 17
        y = group_cfg.y + 17

        test_results_file = os.path.join(
            test_root,
            group_name,
            "test_results.csv"
        )

        if not os.path.exists(test_results_file):
            logging.warning(f"{group_name} missing test_results.csv, skipped.")
            continue

        try:
            test_df = load_test_results(test_results_file)
        except IOError as e:
            logging.error(e)
            continue

        prob_down = test_df['Probability Class 0'].tolist()
        prob_up = test_df['Probability Class 1'].tolist()
        predictions = list(zip(prob_down, prob_up))

        if x < 0 or y > len(prices):
            logging.error(f"{group_name} index out of range.")
            continue

        selected_dates = dates[x:y]
        selected_prices = prices[x:y]

        selected_predictions = generate_labels(predictions, threshold)

        min_len = min(len(selected_prices), len(selected_predictions))
        selected_prices = selected_prices[:min_len]
        selected_dates = selected_dates[:min_len]
        selected_predictions = selected_predictions[:min_len]

        strategy, up_down = process_strategy(
            selected_prices,
            selected_predictions,
            base_money,
            selected_dates
        )

        save_strategy(
            strategy,
            f"{group_name}_threshold_{threshold}",
            output_dir=output_dir
        )

        print(f"{group_name} completed")

    print("\nAll groups processed successfully.")


if __name__ == "__main__":
    main()