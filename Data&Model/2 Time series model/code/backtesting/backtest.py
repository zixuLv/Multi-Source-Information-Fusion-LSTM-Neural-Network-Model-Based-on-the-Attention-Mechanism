# backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import config as C
import utils as u
import os
from functools import reduce
import random 
import logging

random.seed(C.random_seed)
np.random.seed(C.random_seed)

def len_date(strategy):
    """
    Get the maximum and minimum dates in the policy
    """
    if not strategy:
        raise ValueError("none")
    date_1 = [tt[0] for buy_sell in strategy.values() for tt in buy_sell]
    max_date = max(date_1)
    min_date = min(date_1)
    return max_date, min_date

def strategy_change(strategy):
    """
    Sort the policies by date and return a new dictionary of policies
    """
    if not strategy:
        print("none")
        return {}
    
    # Ensure that the dates are sorted in chronological order
    sorted_strategy = dict(sorted(strategy.items()))
    return sorted_strategy
"""
def trade_loop_back1(strategy, date_array, initial_money=10000, fee_rate=0.001):

    
    #Run a backtest of the trading strategy, calculate performance metrics, and use an all-in buy and sell strategy that includes transaction costs.
   
    # Initialize positions and funds
    money = initial_money
    stock_holdings = {}     
    asset = {}              
    daily_return = {}      
    cumulative_yield = {}  
    total_fees = 0.0        
    
    # Initialize positions and trade history
    for date in date_array:
        for action in strategy.get(date, []):
            stock_name = action[2]
            if stock_name not in stock_holdings:
                stock_holdings[stock_name] = 0
    
    total_return = 1  
    
    for idx, date in enumerate(date_array):
        asset[date] = money
        daily_return[date] = 0
        
        for action in strategy.get(date, []):
            act = action[0].lower()
            stock_name = action[2]
            
            current_price = u.data_acquisition(date, stock_name)
            if current_price == -1:
                logging.warning(f"Unable to retrieve {stock_name} at {date} price，Skip this step")
                continue  
            
            # Get yesterday's price
            if idx == 0:
                previous_price = current_price  
            else:
                previous_date = date_array[idx - 1]
                previous_price = u.data_acquisition(previous_date, stock_name)
                if previous_price == -1:
                    previous_price = current_price  
            
            if act == "buy":
                
                denominator = current_price * (1 + fee_rate)
                if denominator == 0:
                    logging.warning(f"The price is zero; this item cannot be purchased {stock_name} at {date}。")
                    continue
                
                quantity = money // denominator  
                if quantity > 0:
                    total_cost = current_price * quantity
                    fee = total_cost * fee_rate 
                    total_cost_with_fee = total_cost + fee
                    if money >= total_cost_with_fee:
                        money -= total_cost_with_fee
                        stock_holdings[stock_name] += quantity
                        total_fees += fee
                        logging.info(f"buy {quantity}  {stock_name} at {date}，price：{current_price:.2f}，Totalcost：{total_cost:.2f}，fee：{fee:.2f}")
                    else:
                        logging.warning(f"Don't have enough funds to buy the entire position {stock_name} at {date}。need：{total_cost_with_fee:.2f}，Available funds：{money:.2f}")
                else:
                    logging.warning(f"The price is too high;can't buy anything with the funds I have available. {stock_name} at {date}。")
            
            elif act == "sell":
                
                quantity = stock_holdings.get(stock_name, 0)
                if quantity > 0:
                    revenue = current_price * quantity
                    fee = revenue * fee_rate  
                    revenue_after_fee = revenue - fee
                    money += revenue_after_fee
                    stock_holdings[stock_name] -= quantity
                    total_fees += fee
                    logging.info(f"sell {quantity} {stock_name} at {date}，price：{current_price:.2f}，TotalRevenue：{revenue:.2f}，fee：{fee:.2f}")
                else:
                    logging.warning(f"have no {stock_name} can sell at {date}")
            
            elif act == "keep":
                
                logging.info(f"keep {stock_name} at {date}")
                pass
            
            elif act == 'none':
                
                logging.info(f"no operation {date}。")
                pass
            
            else:
                logging.error(f"Unknown operation type: {act} at {date}，stock: {stock_name}")
        
        # Calculate the total assets for the day
        total_asset = money
        for stock, qty in stock_holdings.items():
            if qty > 0:
                current_price = u.data_acquisition(date, stock)
                if current_price != -1:
                    total_asset += current_price * qty
        asset[date] = total_asset
        
        # Calculate the yield
        if idx > 0:
            previous_asset = asset[date_array[idx - 1]]
            daily_return[date] = (total_asset - previous_asset) / previous_asset
            total_return *= (1 + daily_return[date])
            cumulative_yield[date] = total_return - 1
        else:
            cumulative_yield[date] = 0
    
    
    returns_series = pd.Series(daily_return)
    yield_series = pd.Series(cumulative_yield)
    
    # max_mdd
    rolling_max = yield_series.cummax()
    drawdown = rolling_max - yield_series
    max_drawdown = drawdown.max()
    
    # Annualized return
    num_days = len(yield_series)
    if num_days == 0:
        annualized_return = 0
    else:
        annualized_return = math.pow(total_return, 360 / num_days) - 1
    
    # sharp
    risk_free_rate = 0.0176
    excess_return = annualized_return - risk_free_rate
    returns_std = returns_series.std()
    if returns_std == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = excess_return / (returns_std * math.sqrt(360))
    
    # risky
    if max_drawdown != 0:
        er_ratio = annualized_return / max_drawdown
    else:
        er_ratio = np.nan
    
    # VaR
    var1 = np.percentile(returns_series.dropna(), 1)
    var5 = np.percentile(returns_series.dropna(), 5)
    
    logging.info(f"done：returns={yield_series.iloc[-1]:.2%}, sharpe_ratio={sharpe_ratio:.4f}, max_mdd={max_drawdown:.2%}, "
                 f"Annualized return={annualized_return:.2%}, final_money={money:.2f}, risky={er_ratio:.4f}, "
                 f"1% VaR={var1:.4f}, 5% VaR={var5:.4f}, cum_fee={total_fees:.2f}")
    
    return yield_series, sharpe_ratio, max_drawdown, annualized_return, money, er_ratio, var1, var5

"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def trade_loop_back2(strategy, date_array, initial_money=10000):
    
    #Run a backtest of the trading strategy, calculate performance metrics, and use an all-in buy and sell strategy。
    
    money = initial_money
    stock_holdings = {}     
    asset = {}             
    daily_return = {}       
    cumulative_yield = {}  
    
  
    for date in date_array:
        for action in strategy.get(date, []):
            stock_name = action[2]
            if stock_name not in stock_holdings:
                stock_holdings[stock_name] = 0
    
    total_return = 1  
    
    for idx, date in enumerate(date_array):
        asset[date] = money
        daily_return[date] = 0
        
        for action in strategy.get(date, []):
            act = action[0].lower()
            stock_name = action[2]
            print('cccccccccc')
            current_price = u.data_acquisition(date, stock_name)
            print(current_price)
            if current_price == -1:
                logging.warning(f"Unable to retrieve {stock_name} at {date} price")
                continue  
            
           
            if idx == 0:
                previous_price = current_price  
            else:
                previous_date = date_array[idx - 1]
                previous_price = u.data_acquisition(previous_date, stock_name)
                if previous_price == -1:
                    previous_price = current_price  
            
            if act == "buy":
                
                quantity = money // current_price  
                if quantity > 0:
                    total_cost = current_price * quantity
                    money -= total_cost
                    stock_holdings[stock_name] += quantity
                    logging.info(f"buy {quantity}  {stock_name} at {date}，price：{current_price}，totalcost：{total_cost}")
                else:
                    logging.info(f"don't have enough money to buy {stock_name} at {date}。")
            
            elif act == "sell":
                
                quantity = stock_holdings.get(stock_name, 0)
                if quantity > 0:
                    revenue = current_price * quantity
                    money += revenue
                    stock_holdings[stock_name] -= quantity
                    logging.info(f"sell {quantity}  {stock_name} at {date}，price：{current_price}，totalrevenue：{revenue}")
                else:
                    logging.info(f"No holdings {stock_name} can sell {date}。")
            
            elif act == "keep":
             
                logging.info(f"keep {stock_name} at {date}")
                pass
            
            elif act == 'none':
                
                logging.info(f"no operation {date}。")
                pass
            
            else:
                logging.error(f"Unknown operation type: {act} at {date}，stock: {stock_name}")
        
        
        total_asset = money
        for stock, qty in stock_holdings.items():
            if qty > 0:
                current_price = u.data_acquisition(date, stock)
                if current_price != -1:
                    total_asset += current_price * qty
        asset[date] = total_asset
        
        
        if idx > 0:
            previous_asset = asset[date_array[idx - 1]]
            daily_return[date] = (total_asset - previous_asset) / previous_asset
            total_return *= (1 + daily_return[date])
            cumulative_yield[date] = total_return - 1
        else:
            cumulative_yield[date] = 0
    
   
    returns_series = pd.Series(daily_return)
    yield_series = pd.Series(cumulative_yield)
    
    
    rolling_max = yield_series.cummax()
    drawdown = rolling_max - yield_series
    max_drawdown = drawdown.max()
    
    
    num_days = len(yield_series)
    if num_days == 0:
        annualized_return = 0
    else:
        annualized_return = math.pow(total_return, 360 / num_days) - 1
    
    
    risk_free_rate = 0.0176
    excess_return = annualized_return - risk_free_rate
    returns_std = returns_series.std()
    if returns_std == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = excess_return / (returns_std * math.sqrt(360))
    
    
    if max_drawdown != 0:
        er_ratio = annualized_return / max_drawdown
    else:
        er_ratio = np.nan
    
    
    var1 = np.percentile(returns_series.dropna(), 1)
    var5 = np.percentile(returns_series.dropna(), 5)
    
    logging.info(f"done：returns={yield_series.iloc[-1]:.2%}, sharp={sharpe_ratio:.4f}, max_mdd={max_drawdown:.2%}, "
                 f"annualized_return={annualized_return:.2%}, final_money={money:.2f}, risky={er_ratio:.4f}, "
                 f"1% VaR={var1:.4f}, 5% VaR={var5:.4f}")
    
    return yield_series, sharpe_ratio, max_drawdown, annualized_return, money, er_ratio, var1, var5

    
def main():
    
    os.makedirs(C.result_plot_dir, exist_ok=True)
    
    # List of Policy Files
    strategy_names = [
        'no_threshold_AR1forecast_baseline','no_threshold_AR2forecast_baseline','no_threshold_ARMA11forecast_baseline','no_threshold_ARMA22forecast_baseline','no_threshold_AR1Archforecast_baseline','no_threshold_AR2Archforecast_baseline','no_threshold_AR1Garchforecast_baseline','no_threshold_AR2Garchforecast_baseline',
        'no_threshold_AR1forecast_beforeevent1','no_threshold_AR2forecast_beforeevent1','no_threshold_ARMA11forecast_beforeevent1','no_threshold_ARMA22forecast_beforeevent1','no_threshold_AR1Archforecast_beforeevent1','no_threshold_AR2Archforecast_beforeevent1','no_threshold_AR1Garchforecast_beforeevent1','no_threshold_AR2Garchforecast_beforeevent1',
        'no_threshold_AR1forecast_afterevent1','no_threshold_AR2forecast_afterevent1','no_threshold_ARMA11forecast_afterevent1','no_threshold_ARMA22forecast_afterevent1','no_threshold_AR1Archforecast_afterevent1','no_threshold_AR2Archforecast_afterevent1','no_threshold_AR1Garchforecast_afterevent1','no_threshold_AR2Garchforecast_afterevent1',
        'no_threshold_AR1forecast_beforeevent2','no_threshold_AR2forecast_beforeevent2','no_threshold_ARMA11forecast_beforeevent2','no_threshold_ARMA22forecast_beforeevent2','no_threshold_AR1Archforecast_beforeevent2','no_threshold_AR2Archforecast_beforeevent2','no_threshold_AR1Garchforecast_beforeevent2','no_threshold_AR2Garchforecast_beforeevent2',
        'no_threshold_AR1forecast_afterevent2','no_threshold_AR2forecast_afterevent2','no_threshold_ARMA11forecast_afterevent2','no_threshold_ARMA22forecast_afterevent2','no_threshold_AR1Archforecast_afterevent2','no_threshold_AR2Archforecast_afterevent2','no_threshold_AR1Garchforecast_afterevent2','no_threshold_AR2Garchforecast_afterevent2',
        'threshold_AR1forecast_baseline','threshold_AR2forecast_baseline','threshold_ARMA11forecast_baseline','threshold_ARMA22forecast_baseline','threshold_AR1Archforecast_baseline','threshold_AR2Archforecast_baseline','threshold_AR1Garchforecast_baseline','threshold_AR2Garchforecast_baseline',
        'threshold_AR1forecast_beforeevent1','threshold_AR2forecast_beforeevent1','threshold_ARMA11forecast_beforeevent1','threshold_ARMA22forecast_beforeevent1','threshold_AR1Archforecast_beforeevent1','threshold_AR2Archforecast_beforeevent1','threshold_AR1Garchforecast_beforeevent1','threshold_AR2Garchforecast_beforeevent1',
        'threshold_AR1forecast_afterevent1','threshold_AR2forecast_afterevent1','threshold_ARMA11forecast_afterevent1','threshold_ARMA22forecast_afterevent1','threshold_AR1Archforecast_afterevent1','threshold_AR2Archforecast_afterevent1','threshold_AR1Garchforecast_afterevent1','threshold_AR2Garchforecast_afterevent1',
        'threshold_AR1forecast_beforeevent2','threshold_AR2forecast_beforeevent2','threshold_ARMA11forecast_beforeevent2','threshold_ARMA22forecast_beforeevent2','threshold_AR1Archforecast_beforeevent2','threshold_AR2Archforecast_beforeevent2','threshold_AR1Garchforecast_beforeevent2','threshold_AR2Garchforecast_beforeevent2',
        'threshold_AR1forecast_afterevent2','threshold_AR2forecast_afterevent2','threshold_ARMA11forecast_afterevent2','threshold_ARMA22forecast_afterevent2','threshold_AR1Archforecast_afterevent2','threshold_AR2Archforecast_afterevent2','threshold_AR1Garchforecast_afterevent2','threshold_AR2Garchforecast_afterevent2'
    ]
    
 
    results = []
    
    for strategy_name in strategy_names:
        strategy_file = os.path.join(C.strategy_dir, f"{strategy_name}.txt")
        try:
           
            strategy = u.load_strategy(strategy_file)
            if not strategy:
                print(f"策略 {strategy_name} 加载失败或为空。")
                continue
            
            
            date_s = strategy_change(strategy)
            print(f"策略 {strategy_name} 重组后的交易操作:", date_s, "\n")

            data_handler = u.DataHandle(date_s)
            date_array = data_handler.date_array

            # Run backtest
            yield_series, sharpe_ratio, max_drawdown, annualized_return, final_money, er_ratio, var1, var5 = trade_loop_back1(strategy, date_array, initial_money=10000000, fee_rate=0.001)#fee
            #yield_series, sharpe_ratio, max_drawdown, annualized_return, final_money, er_ratio, var1, var5 = trade_loop_back2(strategy, date_array, initial_money=10000000)#no fee
            
            # Visualization
            u.plot_profit(yield_series, strategy_name, C.result_plot_dir)
            
            result = {
                "strategy_name": strategy_name,
                "Returncum": f"{yield_series.iloc[-1]:.2%}",
                "Sharpe": f"{sharpe_ratio:.4f}",
                "MDD": f"{max_drawdown:.2%}",
                "annualized_return": f"{annualized_return:.2%}",
                "final_money": f"{final_money:.2f}",
                "risky": f"{er_ratio:.4f}",
                "1% VaR": f"{var1:.4f}",
                "5% VaR": f"{var5:.4f}"
            }
            results.append(result)
            
            
            print(f"strategy: {strategy_name}")
            for key, value in result.items():
                print(f"{key}: {value}")
            print("\n")
        
        except Exception as e:
            print(f"Handling Strategy {strategy_name} an error occurred: {e}\n")
            continue
    
    # Convert the results to a DataFrame and save them to a CSV file
    try:
        results_df = pd.DataFrame(results)
        #csv_path = os.path.join(C.result_csv, "backtest_results_fee.csv")
        csv_path = os.path.join(C.result_csv, "backtest_results.csv")
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"The results of all strategies have been saved to {csv_path}")
    except Exception as e:
        print(f"An error occurred while saving the results to a CSV file: {e}")
            

if __name__ == "__main__":
    main()
