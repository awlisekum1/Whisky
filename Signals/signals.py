#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:28:57 2017

@author: TaoLuo
"""

import pandas as pd
import numpy as np
import pickle
import copy
import datetime
import os
import matplotlib.pyplot as plt
#import statsmodel.api as sm

class Signals:
    def __init__(self):
        pass
#        self.data = copy.deepcopy(mytable)
        
    def calcRHS(self):
#        self.data['delta_idx'] = (self.data['TimeStamp'] - self.data['TimeStamp'].shift(1))/50000000
#        self.data['TradeSize'] = 1
        pass

    def getRHS(self):
        pass
#        self.rhs = rhs
    
    def printSignal(self):
        pass
    
#    def fit(self, data):
#        results = sm.OLS(self.lhs, self.rhs).fit()
#        print(results.summary())
#        return results.params, results.rsquared, results.tvalues
    
    def test(self, coeffs, data):
        pass
    
if __name__ == '__main__':
    pass
        
        
        
        
        
        