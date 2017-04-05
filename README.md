# ML4Trading
A collection of Python programs I wrote to use Machine Learning to do stock market analysis, as part of the CS7646 course at Georgia Tech.

* mc1p1_assess_portfolio: Performs statistical analysis of a diversified stock portfolio and graphs the results. Computes daily average rate of return, standard deviation of daily return, average period return, Sharpe Ratio, and portfolio end value. Plots and saves a graph into the Output folder. Leverages the Pandas and Numpy Python packages.

* mc3p1_assess_learners: Uses Random Trees and Bootstrap Aggregation (Bagging) to conduct regression analysis of certain stock indices. The report summarizes the performance comparison between a Linear Regression, Random Tree, and Bagging on Random Trees.

* mc2p1_marketsim: Simulates orders placed on the stock market and keeps track of the portfolio's value. This program summarizes the portfolio performance on the day of the last order by calculating the Sharpe Ratio, cumulative return, standard deviation, and average daily return of the porfolio and compares it to the S&P 500. I also implemented a process to limit overleveraging. Orders that would cause the portfolio to be overleveraged will not go through.

* mc3h1_defeat_learners: Explores and generates data sets that demonstrate the strengths and weaknesses of Linear Regression and Random Forests. 

* mc3p2_qlearning_robot: Python implementation of Q-Learning algorithm with Dyna. Q-Learning is a reinforcement learning algorithm that  works by running iterative experiments (and, in my case, simulations) based on a reward system to determine the optimal solution. For this project, I implemented Q-Learning to determine the best path out of a given maze design.
