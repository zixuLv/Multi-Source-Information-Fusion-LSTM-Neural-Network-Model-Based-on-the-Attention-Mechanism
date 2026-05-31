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

# 全局变量缓存BTC数据，以避免重复读取
_btc_data_cache = None

def date2num(date_str):
    """
    将日期字符串转换为整数格式 yyyymmdd。
    """
    try:
        return int(pd.to_datetime(date_str).strftime('%Y%m%d'))
    except ValueError:
        raise ValueError("日期格式错误，请使用 'YYYY-MM-DD' 或类似格式")

def num2date(date_num):
    """
    将整数格式 yyyymmdd 转换为日期字符串 'YYYY-MM-DD'。
    """
    year = date_num // 10000
    month = (date_num % 10000) // 100
    day = date_num % 100
    return f"{year}-{month:02d}-{day:02d}"

def load_strategy(file_path):
    """
    从文本文件加载策略。
    文件内容格式示例：
    btc     [[20200203, 'buy', 1], [20200204, 'keep', 0], ..., [20200712, 'none', 0]]
    """
    strategy = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 假设资产名称和操作列表之间由至少一个空格或制表符分隔
                parts = line.strip().split(None, 1)  # 按第一个空白字符分隔
                if len(parts) != 2:
                    print(f"跳过格式不正确的行: {line.strip()}")
                    continue  # 跳过格式不正确的行
                stock, actions_str = parts
                try:
                    # 使用 ast.literal_eval 安全地解析操作列表
                    actions = ast.literal_eval(actions_str)
                    if not isinstance(actions, list):
                        print(f"操作列表格式不正确: {actions_str}")
                        continue
                    for action in actions:
                        if not (isinstance(action, list) and len(action) == 3):
                            print(f"跳过格式不正确的动作: {action}")
                            continue  # 跳过格式不正确的动作
                        date, act, qty = action
                        if not isinstance(date, int):
                            print(f"跳过日期格式不正确的动作: {action}")
                            continue  # 跳过日期格式不正确的动作
                        if date not in strategy:
                            strategy[date] = []
                        strategy[date].append([act, qty, stock])
                except (SyntaxError, ValueError) as e:
                    print(f"解析操作列表失败: {e}，内容: {actions_str}")
                    continue
    except FileNotFoundError:
        print(f"策略文件未找到: {file_path}")
    except Exception as e:
        print(f"加载策略文件失败: {e}")
    return strategy

def data_acquisition(time, data_name):
    """
    从CSV文件中获取特定日期和资产的价格。
    """
    global _btc_data_cache
    if _btc_data_cache is None:
        try:
            _btc_data_cache = pd.read_csv(C.data_paths["btc_data"])
            # print(_btc_data_cache)
            print(f"成功加载 BTC 数据，包含 {len(_btc_data_cache)} 条记录。")
            # 创建 'Date_Num' 列
            if 'Date_Num' not in _btc_data_cache.columns:
                if 'Timestamp' in _btc_data_cache.columns:
                    _btc_data_cache['Date_Num'] = _btc_data_cache['Timestamp'].apply(lambda x: date2num(x))
                    print("已根据 'Timestamp' 列创建 'Date_Num' 列。")
                else:
                    print("BTC数据文件缺少 'Date_Num' 和 'Timestamp' 列。")
                    return -1
            if 'Weighted_Price' not in _btc_data_cache.columns:
                print("BTC数据文件缺少 'Weighted_Price' 列。")
                return -1
        except FileNotFoundError:
            print(f"BTC数据文件未找到: {C.data_paths['btc_data']}")
            return -1
        except Exception as e:
            print(f"加载BTC数据失败: {e}")
            return -1
    
    # 筛选特定日期
    
    filtered_data = _btc_data_cache[_btc_data_cache['Date_Num'] == time]

    if filtered_data.empty:
        print(f"日期不匹配:\n\n{data_name}: {time} 不存在")
        return -1
    else:
        price = filtered_data['Weighted_Price'].values[0]
        print(f"获取到 {data_name} 在 {time} 的价格: {price}")
        return price

def dict2csv(dic, filename):
    """
    将字典写入CSV文件，要求字典的值长度一致。
    """
    try:
        with open(filename, 'w', encoding='utf-8', newline='') as file:
            csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
            csv_writer.writeheader()
            for i in range(len(dic[list(dic.keys())[0]])):
                row = {key: dic[key][i] for key in dic.keys()}
                csv_writer.writerow(row)
    except Exception as e:
        print(f"写入CSV文件失败: {e}")

def plot_profit(yield_series, strategy_name, save_dir):
    """
    绘制累计收益曲线并保存为PNG文件。
    """
    try:
        dates = [num2date(day) for day in sorted(yield_series.keys())]
        profits = [yield_series[day] for day in sorted(yield_series.keys())]

        # 调整字体大小
        plt.rcParams.update({'font.size': 16})  # 全局字体大小

        plt.figure(figsize=(12, 9))
        plt.plot(profits, color='purple', label='Strategy',linewidth=2.5)

        # 添加字体大小参数
        plt.xlabel("Date", fontsize=20)  # X轴标签字体
        plt.ylabel("Accumulated Profit", fontsize=20)  # Y轴标签字体
        plt.title(f"Cumulative Profit: {strategy_name}", fontsize=18)  # 标题字体

        # 设置X轴刻度
        if len(profits) > 10:
            ticks = np.linspace(0, len(profits) - 1, 10).astype(int)
        else:
            ticks = range(len(profits))
        plt.xticks(ticks=ticks, labels=[dates[i] for i in ticks], rotation=45, fontsize=20)  # 刻度字体

        plt.yticks(fontsize=20)  # Y轴刻度字体
        plt.legend(fontsize=20)  # 图例字体
        plt.tight_layout()

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{strategy_name}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")

class DataHandle:
    """
    处理策略数据，初始化日期、股票、动作和交易量数组。
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
