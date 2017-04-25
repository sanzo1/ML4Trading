"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.threshold = 0


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):
        num_shares = 200
        steps = 3
        self.learner = ql.QLearner(num_states=steps, num_actions=3)
        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices.to_csv("prices.csv")
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        high_all = ut.get_data(syms, dates, colname = "High")  # automatically adds SPY
        high = high_all[syms]  # only portfolio symbols

        low_all = ut.get_data(syms, dates, colname="Low")  # automatically adds SPY
        low = low_all[syms]  # only portfolio symbols

        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols

        span = 3
        SODF = self.calcSO(prices, high, low, syms, span=span)
        SMA = self.calcBBands(prices)
        SMA.to_csv("SMA.csv")
        forceIndex = self.calcForceIndex(prices, volume)

        discretize, self.threshold = self.discretizeIndicators(forceIndex, steps=steps)
        discretize.to_csv("discretize.csv")
        position = 1
        for i in range(0, 200):
            state = discretize.iloc[0]
            action = self.learner.querysetstate(state)
            position, posnum = self.doAction(action, position, day=0)
            for day in range(1, len(discretize)):
                reward = self.getReward(action, prices, day, symbol, num_shares)
                state = discretize.iloc[day]
                #print "state"
                #print state
                #print "prevaction"
                #print action
                #print "reward"
                #print reward
                action = self.learner.query(state, reward)
                #print "newaction"
                #print action
                position, posnum = self.doAction(action, position, day=day)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        num_shares = 200
        steps = 3
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices = prices_all[syms]  # only portfolio symbols
        if self.verbose: print prices

        high_all = ut.get_data(syms, dates, colname="High")  # automatically adds SPY
        high = high_all[syms]  # only portfolio symbols

        low_all = ut.get_data(syms, dates, colname="Low")  # automatically adds SPY
        low = low_all[syms]  # only portfolio symbols

        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        span = 3
        SODF = self.calcSO(prices, high, low, syms, span=span)
        SMA = self.calcBBands(prices)
        forceIndex = self.calcForceIndex(prices, volume)
        discretize = self.discretizeIndicatorsTest(forceIndex, self.threshold, steps=steps)
        position = 1

        df_trades = discretize.copy(deep=True)
        df_trades['FI'] = 0.0

        testlist1 = []
        testlist2 = []
        action = self.learner.querysetstate(discretize.iloc[0])
        position, posnum = self.doAction(action, position, day=0)
        testlist1.append(position)
        testlist2.append(posnum)
        df_trades.iloc[0]['FI'] = posnum
        for day in range(1, len(discretize)-1):
            action = self.learner.querysetstate(discretize.iloc[day])
            position, posnum = self.doAction(action, position, day=day)
            testlist1.append(position)
            testlist2.append(posnum)
            df_trades.iloc[day]['FI'] = posnum
        diff = df_trades.sum(axis=0).values[0]
        df_trades.iloc[len(discretize)-1]['FI'] = diff * -1.0
        df_trades.to_csv("trades.csv")
        #pd.DataFrame(testlist1).to_csv("testlist1.csv")
        #pd.DataFrame(testlist2).to_csv("testlist2.csv")
        return df_trades



    def calcSO(self, prices, high, low, symbols, span=3):

        # get daily portfolio value
        normedprice = prices / prices.ix[0]
        normedhigh = high / high.ix[0]
        normedlow = low / low.ix[0]
        # Stochastic oscillator %K
        SOk = pd.DataFrame(((normedprice[symbols] - pd.rolling_mean(normedlow[symbols], window=span).min()) / (
        pd.rolling_mean(normedhigh[symbols], window=span).max() - pd.rolling_mean(normedlow[symbols], window=span).min())))
        SOk.columns = ["%K"]

        # Stochastic oscillator %D
        SOd = pd.DataFrame(pd.rolling_mean(SOk, window=span))
        SOd.columns = ["%D"]
        SOd.fillna(value=0.0, inplace=True)
        return SOd
        #print SOd
        #SOk.fillna(value=0.0, inplace=True)
        #return SOk

    def calcForceIndex(self, prices, volume):
        normed = prices / prices.ix[0]
        #normvol = volume / volume.ix[0]
        forceIndex = (normed - normed.shift(1)) * volume

        forceIndex.fillna(value=0.0, inplace=True)
        forceIndex.columns = ["FI"]
        return forceIndex


    def calcBBands(self, prices, length=3, numsd=2):
        """ returns average, upper band, and lower band"""
        # get daily portfolio value
        normed = prices / prices.ix[0]
        SMA = pd.DataFrame(pd.rolling_mean(normed, window=length))
        sd = pd.DataFrame(pd.rolling_std(normed, window=length))

        upband = SMA + (sd * numsd)
        dnband = SMA - (sd * numsd)
        upband.fillna(value=0.0, inplace=True)
        dnband.fillna(value=0.0, inplace=True)

        upband.columns = ["UBB"]
        dnband.columns = ["LBB"]
        SMA.columns = ["SMA"]


        return SMA;


    def discretizeIndicators(self, indicator, steps=10):
        stepsize = (len(indicator) / steps)
        min = indicator.min(axis=1, skipna=True)
        max = indicator.max(axis=1, skipna=True)
        sortedind = indicator.sort_values(['FI'], ascending=True)
        threshold = []
        binlabels = range(0, steps-1)

        for i in range(0, steps):
            threshold.append(sortedind.iloc[(i + 1) * stepsize]['FI'])
        threshold[0] = -9999999.0
        threshold[steps-1] = 9999999.0
        indicator.to_csv("testind.csv")
        indicator['FI'] = pd.cut(indicator['FI'], threshold, labels=binlabels, right=True)
        indicator.to_csv("testind2.csv")
        indicator.fillna(value=binlabels[steps-2], inplace=True)
        indicator.to_csv("testind3.csv")
        return indicator, threshold

    def discretizeIndicatorsTest(self, indicator, threshold, steps=10):
        stepsize = (len(indicator) / steps)
        min = indicator.min(axis=1, skipna=True)
        max = indicator.max(axis=1, skipna=True)
        sortedind = indicator.sort_values(['FI'], ascending=True)

        binlabels = range(0, steps-1)

        #for i in range(0, steps):
        #    threshold.append(sortedind.iloc[(i + 1) * stepsize]['%D'])
        indicator['FI'] = pd.cut(indicator['FI'], threshold, labels=binlabels, right=True)
        indicator.fillna(value=binlabels[steps-2], inplace=True)
        return indicator


    def doAction(self, action, position, day):
        posnum = 0
        if action == 0:
            if position == 1:
                position = 0
                posnum = -200
            elif position == 2:
                position = 0
                posnum = -400
            else:
                position = 0
                posnum = 0
        elif action == 2:
            if position == 1:
                position = 2
                posnum = 200
            elif position == 0:
                position = 2
                posnum = 400
            else:
                position = 2
                posnum = 0
        elif action == 1:
            if position == 0:
                position = 1
                posnum = 200
            elif position == 2:
                position = 1
                posnum = -200
            else:
                position = 1
                posnum = 0
        return position, posnum

    def getReward(self, action, prices, day, symbol, numshares=200):
        reward = 0
        prevsharevalue = prices.iloc[day-1][symbol]
        currsharevalue = prices.iloc[day][symbol]
        profit = currsharevalue / prevsharevalue

        # if SHORT
        if action == 0:
            if profit > 1:
                reward = profit * (-100)
                #reward = -100
            elif profit < 1:
                reward = profit * 100
                #reward = 100
            else:
                reward = 0
        # if NOTHING
        elif action == 1:
            reward = -50
        # if LONG
        elif action == 2:
            if profit > 1:
                reward = profit * 100
                #reward = 100
            elif profit < 1:
                reward = profit * (-100)
                #reward = 100
            else:
                reward = 0
        return reward

    def author(self):
        return 'jshi88'  # replace tb34 with your Georgia Tech username.

if __name__=="__main__":
    print "One does not simply think up a strategy"
