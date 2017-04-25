#! /usr/bin/python2.7
"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data, get_all_data
from matplotlib.font_manager import FontProperties

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality

def getSMA(start_date, end_date, symbols, start_val, length=10, gen_plot=True, normalize=True):

    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds spy
    prices = prices_all[symbols]  # only portfolio symbols
    if normalize:
    # get daily portfolio value
        normed = prices / prices.ix[0]
        SMA = normed.rolling(length).mean()
        SMA.columns = ["Moving Average"]
    else:
        SMA = prices.rolling(length).mean()
        SMA.columns = ["Moving Average"]

    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([normed, SMA], axis=1)
        plt = df_temp.plot()

        plt.set_ylabel("normalized price")
        plt.set_xlabel("date")
        plt.set_title("Daily AAPL Value and 10-Day Simple Moving Average")
        fig = plt.get_figure()
        fig.savefig('output/plot.png')

    return SMA

def getBBands(start_date, end_date, symbols, start_val, gen_plot=True, length=10, numsd=2, normalize=True):
    """ returns average, upper band, and lower band"""
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds spy
    prices = prices_all[symbols]  # only portfolio symbols

    # get daily portfolio value
    if normalize:
        normed = prices / prices.ix[0]
        SMA = normed.rolling(length).mean()


        sd = normed.rolling(length).std()

    else:
        SMA = prices.rolling(length).mean()

        sd = prices.rolling(length).std()
    upband = SMA + (sd * numsd)
    dnband = SMA - (sd * numsd)


    upband.columns = ["Upper Bollinger Band"]
    dnband.columns = ["Lower Bollinger Band"]
    SMA.columns = ["10-Day Moving Average"]

    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([normed, SMA, upband, dnband], axis=1)
        plt = df_temp.plot()
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.set_ylabel("normalized price")
        plt.set_xlabel("date")
        plt.set_title("Bollinger Bands for AAPL")
        fig = plt.get_figure()
        fig.savefig('output/plot2.png', bbox_inches='tight')

    return pd.concat([upband, dnband], axis=1);

def getSO(start_date, end_date, symbols, start_val, span, gen_plot=True):

    dates = pd.date_range(start_date, end_date)
    prices_all = get_all_data(symbols, dates)  # automatically adds spy

    # get daily portfolio value
    normed = prices_all / prices_all.ix[0]
    closeonly = normed[['Adj Close']]
    closeonly.columns = [["AAPL"]]
    # Stochastic oscillator %K
    SOk = pd.DataFrame(((normed['Adj Close'] - normed['Low'].rolling(span).min()) / (normed['High'].rolling(span).max() - normed['Low'].rolling(span).min())), columns = ["%K"])
    SOk.columns = [["%K"]]

    # Stochastic oscillator %D
    SOd = SOk.rolling(3).mean()
    SOd.columns = [["%D"]]


    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([closeonly, SOk, SOd], axis=1)

        plt = df_temp.plot()
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.set_ylabel("normalized price")
        plt.set_xlabel("date")
        plt.set_title("Stochastic Oscillators for AAPL")
        plt.set_ylim(0, 2)
        fig = plt.get_figure()
        fig.savefig('output/plot3.png', bbox_inches='tight')

    return pd.concat([SOk, SOd], axis=1)

def assess_portfolio(sd , ed , \
    syms , \
    allocs, \
    sv, rfr=0.0, sf=252.0, \
    gen_plot=False):



    """

    return the cumulative return, average daily return (assuming sf = 252), standard deviation of daily return,
    sharpe ratio, and end value of the portfolio

    @type sd: date
    @param sd: start date

    @type ed: date
    @param ed: end date

    @type syms: list
    @param syms: list of stock ticker symbols to analyze

    @type allocs: list
    @param allocs: proportional allocation of stocks in portfolio

    @type sv: number
    @param sv: starting portfolio value

    @type rfr: number
    @param rfr: risk-free return per sample period

    @type sf: number
    @param sf: sampling frequency per year

    @type gen_plot: boolean
    @param gen_plot: indicator on whether or not to plot the data

    @return: the cumulative return, average period return, standard deviation of daily return, sharpe ratio, end value

    """

    # read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds spy
    prices = prices_all[syms]  # only portfolio symbols
    prices_spy = prices_all['SPY']  # only spy, for comparison later

    # get daily portfolio value
    normed = prices/prices.ix[0]
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis=1)

    # get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(port_val, allocs, rfr, sf) # add code here to compute stats

    # compare daily portfolio value with spy using a normalized plot
    if gen_plot:
        # add code to plot here
        port_val_norm = port_val/port_val.ix[0]
        spy_norm = prices_spy/prices_spy.ix[0]
        df_temp = pd.concat([port_val_norm, spy_norm], keys=['portfolio', 'SPY'], axis=1)
        plt = df_temp.plot()
        plt.set_ylabel("normalized price")
        plt.set_xlabel("date")
        plt.set_title("daily portfolio value and SPY")
        fig = plt.get_figure()
        fig.savefig('output/plot.png')

    # add code here to properly compute end value
    # note: the end value computed here factors in the starting value.
    ev = port_val[-1]

    return cr, adr, sddr, sr, ev

def compute_portfolio_stats(port_val , allocs, rfr, sf):

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
    #normalize and process the data
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]
    cum_ret = (port_val[-1]/port_val[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sr = np.sqrt(sf) * avg_daily_ret / std_daily_ret

    return cum_ret, avg_daily_ret, std_daily_ret, sr


def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    symbols = ['AAPL']
    allocations = [1.0]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    SMA = getSMA(start_date, end_date, symbols, start_val, gen_plot=True)
    BBand = getBBands(start_date, end_date, symbols, start_val)
    BBand.to_csv('BBand.csv')
    SO = getSO(start_date, end_date, symbols, start_val, span = 15, gen_plot=True)
    SO.to_csv('Stochastic.csv')



if __name__ == "__main__":
    test_code()
