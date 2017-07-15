#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:39:35 2017

@author: TaoLuo
"""

import sys
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Signals')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Lhs')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Fitter')
import copy
import numpy as np
import pandas as pd
import fitOLS

class fitterGenerator:
    def __init__(self, param):
        self.__param = param
        self._parseParam()
    
    def _parseParam(self):
        self.fitterName = self.__param['fitterName']
        self.fitterParams = self.__param['fitterParams']
        
    def generateFitter(self):
        if self.fitterName == 'OLS':
            return self.generateOLS
    
    def generateOLS(self, Y, X):
        self.ols = fitOLS.OLSfitter(self.fitterParams)
        return self.ols.fitting(Y, X)
        
# test
#t = fitterGenerator()
#s = t.generateFitter()
#res = s(X, Y)