import config as C
import pandas as pd
import glob
import os
import torch
import numpy as np
import pandas as pd
import utils as U
import logging


def save_strategy(strategy, i, output_dir="output_strategy"):
    """将策略保存到指定文件夹中的文本文件"""
    # 确保目标文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, f"BTC_the_up_down_{i}_baseline.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        for stock, actions in strategy.items():
            f.write(f"{stock}\t{actions}\n")
    logging.info(f"策略已保存到: {file_path}")


def date_to_num(date: str) -> int:
    """
    将日期字符串转换为整数格式。

    Args:
        date (str): 日期字符串，格式为"YYYY/MM/DD"。

    Returns:
        int: 转换后的整数格式日期。

    Raises:
        ValueError: 如果日期格式不正确。
    """
    try:
        year, month, day = map(int, date.split("/"))
    except ValueError:
        raise ValueError("请检查日期格式。示例: 2015/09/10")
    return year * 10000 + month * 100 + day


def process_strategy(f_p, zz, base_money, fdd):
    """处理策略和收益"""
    strategy = {'btc': []}
    up_down = {'lstm': []}
    cash = base_money  # 当前现金
    holdings = 0  # 当前持仓

    for i in range(len(f_p)):
        s = []
        p = []

        # 确保 f_p[i] 是一个浮动类型的数字
        try:
            price = float(f_p[i])  # 将价格转换为浮动类型
        except ValueError:
            logging.error(f"无效的价格数据：{f_p[i]}，跳过该数据。")
            continue  # 如果转换失败，跳过该数据

        if zz[i] == 1:  # 买入信号
            if cash > 0:  # 只有在有现金时才买入
                buy_num = cash / price  # 用所有现金买入
                holdings += buy_num  # 更新持仓
                cash = 0  # 买入后现金清空
                s.append(date_to_num(fdd[i]))
                s.append('buy')
                s.append(1)
                p.append(date_to_num(fdd[i]))
                p.append(1)
                up_down['lstm'].append(p)
                strategy['btc'].append(s)
                
        elif zz[i] == 0:  # 卖出信号
            if holdings > 0:  # 只有在有持仓时才卖出
                sell_num = holdings  # 卖出所有持仓
                cash += sell_num * price  # 卖出后更新现金
                holdings = 0  # 卖出后持仓清空
                s.append(date_to_num(fdd[i]))
                s.append('sell')
                s.append(0)
                p.append(date_to_num(fdd[i]))
                p.append(0)
                up_down['lstm'].append(p)
                strategy['btc'].append(s)

    # 在最后一天强制卖出
    if holdings > 0:  # 如果还有持仓
        sell_num = holdings  # 卖出所有持仓
        final_price = float(f_p[-1])  # 获取最后一天的价格
        cash += sell_num * final_price  # 卖出后更新现金
        holdings = 0  # 持仓清空
        final_date = date_to_num(fdd[-1])  # 最后一天的日期
        strategy['btc'].append([final_date, 'sell', 0])  # 强制卖出操作
        up_down['lstm'].append([final_date, 0])  # 强制卖出记录

    return strategy, up_down


def label_tal(dier: str):
    """加载标签和日期数据"""
    suoyin_data = pd.read_csv(dier)
    label_list = list(map(float, suoyin_data.values[0]))
    date_list = list(suoyin_data.values[1])
    return label_list, date_list


def generate_labels(zz, values):
    """生成买入卖出标签"""
    return [0 if z[0] > values else 1 for z in zz]


def load_test_results(file_path: str) -> pd.DataFrame:
    """
    从CSV文件中读取测试结果数据。

    Args:
        file_path (str): CSV文件路径。

    Returns:
        pd.DataFrame: 包含测试结果的数据框。

    Raises:
        IOError: 如果文件读取失败。
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = {'True Label', 'Predicted Label', 'Probability Class 0', 'Probability Class 1'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"缺少必要的列: {missing}")
        return df
    except Exception as e:
        raise IOError(f"读取文件失败: {e}")


def load_price_data(file_path: str) -> pd.DataFrame:
    """
    读取价格数据文件，返回日期和价格的数据框。
    
    Args:
        file_path (str): 价格数据文件路径。

    Returns:
        pd.DataFrame: 包含日期和价格的数据框。
    """
    try:
        # 读取 CSV 文件，假设文件中没有表头，日期和价格分别在第一行和第二行
        df = pd.read_csv(file_path, header=None)

        # 检查文件格式是否正确
        if df.shape[0] < 2:
            raise ValueError(f"文件 {file_path} 格式不正确，数据不足。")

        # 获取日期和价格
        dates = df.iloc[0].tolist()  # 第一行是日期
        prices = df.iloc[1].tolist()  # 第二行是价格

        # 构建数据框并返回
        price_df = pd.DataFrame({
            'date': dates,
            'price': prices
        })

        return price_df
    except Exception as e:
        logging.error(f"读取价格数据失败: {e}")
        raise


def main():
    base_money = 10000000.0  # 可以根据需要调整

    # 数据文件路径
    price_data_file = r"data\suoyin.csv"  # 替换为您的价格数据CSV文件路径
    test_results_file = r"test_results\test_results.csv"  # 替换为您的测试结果CSV文件路径
    x = U.x + 17
    y = U.y + 17
    
    if not os.path.exists(price_data_file):
        logging.error(f"价格数据文件不存在: {price_data_file}")
        return
    if not os.path.exists(test_results_file):
        logging.error(f"测试结果文件不存在: {test_results_file}")
        return

    # 读取价格数据
    try:
        price_df = load_price_data(price_data_file)
    except IOError as e:
        logging.error(e)
        return

    # 读取测试结果数据
    try:
        test_df = load_test_results(test_results_file)
    except IOError as e:
        logging.error(e)
        return

    # 提取必要的列
    dates = price_df['date'].tolist()
    prices = price_df['price'].tolist()

    prob_down = test_df['Probability Class 0'].tolist()
    prob_up = test_df['Probability Class 1'].tolist()
    predictions = list(zip(prob_down, prob_up))

    # 检查索引是否在范围内
    if x < 0 or y > len(prices):
        logging.error("索引x或y超出价格数据范围。")
        return

    # 根据区间获取dates和prices
    selected_dates = dates[x:y]
    selected_prices = prices[x:y]

    selected_predictions = predictions
    selected_predictions = generate_labels(selected_predictions, 0.5)

    # 策略和收益计算
    strategy, up_down = process_strategy(selected_prices, selected_predictions, base_money, selected_dates)

    # 保存策略到文件夹
    save_strategy(strategy, 'lstm', output_dir="output_strategy")


if __name__ == "__main__":
    main()