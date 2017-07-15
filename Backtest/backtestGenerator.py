#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:04:59 2017

@author: TaoLuo
"""

import sys
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Signals/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Lhs/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Fitter/')
import copy
import numpy as np
import pandas as pd
import simpleBacktester as simpleBT
import commonBacktester as commonBT
import random

class backtestGenerator:
    def __init__(self, param):
        self.__param = param
        self._parseParam()
        
    def _parseParam(self):
        self.backtesterName = self.__param['backtesterName']
        
    def generateBacktester(self):
        if self.backtesterName == 'simpleBacktester':
            return self.generateSimpleBacktester
        if self.backtesterName == 'commonBacktester':
            return self.generateCommonBacktester
    
    def generateSimpleBacktester(self, bookData, fitter, signals = ['OIR','VOI','macd', 'EMA', 'WILLR']):
        self.BTester = simpleBT.simpleBacktester(self.__param)
        return self.BTester.backtesting(bookData, fitter, signals)
        
    def generateCommonBacktester(self, bookData, fitter, signals = ['OIR','VOI','macd', 'EMA', 'WILLR']):
        self.BTester = commonBT.commonBacktester(self.__param)
        return self.BTester.backtesting(bookData, fitter, signals)    