#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:42:27 2017

@author: TaoLuo
"""

import sys
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Signals')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Lhs')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Fitter')
import copy
import numpy as np
import pandas as pd
from fitter import fitter
from sklearn import linear_model
import statsmodels.api as sm

class OLSfitter(fitter):
    def __init__(self, param = 1):
        fitter.__init__(self)
        self.__param = param
    
    def fitting(self, Y, X):
        self.reg = sm.OLS(Y,X)
        return self.reg.fit()