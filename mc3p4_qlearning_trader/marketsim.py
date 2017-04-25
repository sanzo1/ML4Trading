"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'jshi88'  # replace tb34 with your Georgia Tech username.

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    """

    The main function for simulating the market

    @type orders_file: string
    @param orders_file: location of file with orders

    @type start_val: int
    @param start_val: starting cash value of portfolio

    @return: returns a dataframe with portfolio values

    """

    #read in stock orders, sort them in chronological order
    stock_orders = pd.read_csv(orders_file, index_col=0, parse_dates=True)
    stock_orders.sort_index()

    #extract the symbol of stock orders and put them in a list
    symbols = stock_orders['Symbol'].unique().tolist()

    #calculate start and end dates for orders (date of first order and ate of last order)
    begin_date = stock_orders.index[0].to_datetime()
    end_date = stock_orders.index[-1].to_datetime()

    #gets all the days between beginning and end date
    date = pd.date_range(start=begin_date, end=end_date)

    #filters only the days that S&P 500 trades
    date_range = get_data(['$SPX'], date).index.get_values().astype('M8[D]')

    #add CASH to the list of symbols
    full_portfolio = list(symbols) + ['CASH']

    #get the daily trading information for all the symbols
    port_prices = get_data(symbols, date_range)[symbols]

     #initialize the data frame that tracks daily portfolio values
    portfolio_tracker = pd.DataFrame(data=np.zeros((len(date_range), len(symbols) + 1)), columns=full_portfolio, index=[date_range])
    portfolio_tracker.iloc[0]['CASH'] = start_val

    #iterate through each order and process them, updating the portfolio_tracker data frame in each iteration
    for i in range(len(stock_orders)):
        #check for overleverage, if it's false, proceed.
        if check_leverage(i, portfolio_tracker, stock_orders, port_prices, symbols) == False:
             #if the order is a "SELL", we use a negative multiplier for subtraction later on
            if stock_orders.ix[i]['Order'].lower() == "sell":
                mult_factor = -1.0
            else:
                mult_factor = 1.0

            #update transaction for stock symbol
            new_stock_shares = portfolio_tracker.ix[stock_orders.index[i].to_datetime(), stock_orders.ix[i]['Symbol']] + mult_factor * stock_orders.ix[i]['Shares']
            portfolio_tracker.ix[stock_orders.index[i].to_datetime(), stock_orders.ix[i]['Symbol']] = new_stock_shares
            #calculate the total amount of cash gained/lost for the order, and update the CASH value
            stock_price = port_prices.ix[stock_orders.index[i].to_datetime()][stock_orders.ix[i]['Symbol']]
            total_order_cost = stock_orders.ix[i, 'Shares'] * stock_price * mult_factor
            portfolio_tracker.ix[stock_orders.index[i].to_datetime(), 'CASH'] = portfolio_tracker.ix[stock_orders.index[i].to_datetime(), 'CASH'] - total_order_cost

    #fill the rest of the CASH columns to ensure that portfolio_tracker keeps track of the portfolio every day
    portfolio_tracker['CASH'] = portfolio_tracker['CASH'].cumsum()

    #fill the rest of the Symbol columns to ensure that portfolio_tracker keeps track of the portfolio every day
    portfolio_tracker[list(symbols)] = portfolio_tracker[list(symbols)].cumsum()

    #sum up all the values and return as a DataFrame
    portfolio_tracker[list(symbols)] = portfolio_tracker[list(symbols)] * port_prices
    port_val = portfolio_tracker.sum(axis=1)
    return pd.DataFrame(data=port_val)

def check_leverage(i, portfolio_tracker, stock_orders, port_prices, symbols):
    """

    Calculates leverage and returns a boolean value on whether or not the account is over-leveraged or not

    @type i: int
    @param i: current iteration in stock orders

    @type portfolio_tracker: DataFrame
    @param portfolio_tracker: tracks portfolio values for each day

    @type stock_orders: DataFrame
    @param stock_orders: all orders that were read in

    @type port_prices: DataFrame
    @param port_prices: Stock value prices for symbols and date ranges

    @return: returns a boolean value on whether or not the account is over-leveraged or not

    """

    #This function works by making a simulated trade and then calculating the leverage. This is a pretty naive and
    #computationally exhaustive way because of duplicative efforts. If I had more time, I would have found a way to optimize
    #this section
    is_over_leveraged = False
    leverage_limit = 1.5

    port_copy = portfolio_tracker.copy(deep=True)
    orders_copy = stock_orders.copy(deep=True)

    if orders_copy.ix[i]['Order'].lower() == "sell":
        mult_factor = -1.0
    else:
        mult_factor = 1.0

    new_stock_shares = port_copy.ix[
                           orders_copy.index[i].to_datetime(), orders_copy.ix[i]['Symbol']] + mult_factor * \
                                                                                              orders_copy.ix[i][
                                                                                                    'Shares']
    port_copy.ix[orders_copy.index[i].to_datetime(), orders_copy.ix[i]['Symbol']] = new_stock_shares
    # calculate the total amount of cash gained/lost for the order, and update the CASH value
    stock_price = port_prices.ix[orders_copy.index[i].to_datetime()][orders_copy.ix[i]['Symbol']]
    total_order_cost = orders_copy.ix[i, 'Shares'] * stock_price * mult_factor
    port_copy.ix[orders_copy.index[i].to_datetime(), 'CASH'] = port_copy.ix[orders_copy.index[i].to_datetime(), 'CASH'] - total_order_cost

    port_copy['CASH'] = port_copy['CASH'].cumsum()
    port_copy[list(symbols)] = port_copy[list(symbols)].cumsum()
    port_copy[list(symbols)] = port_copy[list(symbols)] * port_prices
    port_sum_with_cash = port_copy.ix[orders_copy.index[i].to_datetime()]
    port_sum_with_cash = port_sum_with_cash.sum(axis=0)

    port_sum_minus_cash = port_copy.ix[orders_copy.index[i].to_datetime(), :-1]
    port_sum_minus_cash = port_sum_minus_cash.abs()
    port_sum_minus_cash = port_sum_minus_cash.sum(axis=0)

    calc_leverage = (port_sum_minus_cash) / (port_sum_with_cash)

    if calc_leverage > leverage_limit:
        is_over_leveraged = True
    return is_over_leveraged

def compute_portfolio_stats(port_val, rfr, sf):
    """

    Return the cumulative return, average daily return, standard deviation of daily return, and Sharpe Ratio

    @type port_val: Pandas Data Frame
    @param port_val: Portfolio data to analyze

    @type allocs: list
    @param allocs: proportional allocation of stocks in portfolio

    @type rfr: number
    @param rfr: Risk-free return per sample period

    @type sf: number
    @param sf: Sampling frequency per year

    @return: the cumulative return, average period return, standard deviation of daily return, sharpe ratio

    """
    # normalize and process the data
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]
    cum_ret = (port_val[-1] / port_val[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sr = np.sqrt(sf) * avg_daily_ret / std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sr

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders3.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        print "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    print
    start_date = portvals.index[0]

    end_date = portvals.index[-1]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals, rfr=0.0, sf=252.0)
    spyvals = get_data(['$SPX'], dates = pd.date_range(start_date, end_date))

    spyvals = spyvals[spyvals.columns[1]]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(spyvals, rfr=0.0, sf=252.0)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()