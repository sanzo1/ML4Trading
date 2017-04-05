"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    X = np.random.normal(loc=5, scale=10, size=(1000, 2))
    Y = np.random.normal(loc=5, scale=1, size=1000)
    return X, Y

def best4RT(seed=1489683273):
    np.random.seed(seed)
    X = np.random.random(size=(1000, 800))
    Y = np.random.random(size=1000)
    return X, Y

if __name__=="__main__":
    print "they call me Tim."