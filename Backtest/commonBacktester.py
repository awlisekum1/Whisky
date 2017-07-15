#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:20:14 2017

@author: TaoLuo
"""


import sys
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Data')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Signals/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Lhs/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Fitter/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Backtest/')
import copy
import numpy as np
import pandas as pd
import random
from backtest import Backtester

class commonBacktester(Backtester):
    def __init__(self, param):
        Backtester.__init__(self)
        self.__param = param
        self._parseParam()

    def _parseParam(self):
        self.weights = self.__param['weight']
        self.nWeights = len(self.weights)
        self.threshold = self.__param['threshold']
        self.cost = self.__param['cost']
        self.sample = self.__param['sample']
        self.threshold = self.__param['threshold']
    
    def backtesting(self, bookData, fitter, signals = ['OIR','VOI','macd', 'EMA', 'WILLR']):
        self.trainContracts = bookData.trainSet
        self.nTrainContracts = len(bookData.trainSet)

        self.testContracts = bookData.testSet
        self.nTestContracts = len(bookData.testSet)
        
        self.signals = signals
        self.fitting(bookData, fitter, signals)
        self.predicting(bookData, signals)
        self.orderLogic(bookData, self.cost)
        self.summary()
        pass
    
    def fitting(self, bookData, fitter, signals):
        # param1:
        #1 fit the whole day data
        #2 fit single session data
        #3 fit the insample data
        
        # param2:
        # fit number of days/sessions
        
        # if fit the whole day data
        sigTStats = ['t_%s' % s for s in self.signals]
        sigCoeffs = ['coefs_%s' % s for s in self.signals]
        self.InSampleStats = pd.DataFrame(columns = ['ISR2', 'ISR2ADJ'] + sigTStats + sigCoeffs)

        regData = pd.DataFrame(None)
        for contract in self.trainContracts:
            for i in range(len(bookData.tradingSession)):
                regData = regData.append(bookData.dataSignal[contract][i])
        self.regressors = fitter(regData['lhs'], regData[signals])
            
        ISR2 = self.regressors.rsquared
        ISR2ADJ = self.regressors.rsquared_adj
        tstat = self.regressors.tvalues.values
        coefs = self.regressors.params.values
        self.InSampleStats.loc['train'] = np.r_[ISR2, ISR2ADJ, tstat, coefs]
        pass
    
    def predicting(self, bookData, signals):
        # param1:
        #1 rolling prediction
        #2 prediction for out of sample
        self.prediction = {}
        self.response = {}
        for iDay, contract in enumerate(self.testContracts):
            predictData = pd.DataFrame(None)
            for i in range(len(bookData.tradingSession)):
                predictData = predictData.append(bookData.dataSignal[contract][i])

            self.response[iDay] = predictData['lhs']
            self.prediction[iDay] = self.regressors.predict(predictData[signals])  
            
        pass
    
    def orderLogic(self, bookData, cost): 
        self.pnl = [0]
        self.position = [0]
        maxpos = 1     
        
        self.OutOfSampleStats = pd.DataFrame(columns = ['PnL', 'Volume','OSR2','OSR2ADJ'])
        for iDay, contract in enumerate(self.testContracts):
            print(iDay, contract)
            bank_balance0, bank_balance = self.pnl[-1], self.pnl[-1]

            y = self.response[iDay]
            y_hat = self.prediction[iDay]
            y_bar = np.mean(y)
            tss  = y - y_bar
            TSS = np.dot(tss,tss)
            rss = y_hat - y
            RSS = np.dot(rss,rss)
            ess = y_hat - y_bar
            ESS = np.dot(ess,ess)
            OSR2 = 1 - RSS/TSS
            OSR2ADJ = 1 - RSS/TSS*(len(y) - 1)/(len(y) - len(self.signals)-1)
            
            pos = 0
            volume = 0
            pnl = []
            position = []
            orderData = pd.DataFrame(None)
            for i in range(len(bookData.tradingSession)):
                orderData = orderData.append(bookData.dataCleaned[contract][i][['AskPrice1', 'BidPrice1']])
            askPrice = orderData.AskPrice1.values
            bidPrice = orderData.BidPrice1.values
            for k in range(len(self.prediction[iDay])):
                trade = 0
                if self.prediction[iDay][k] > self.threshold and pos < maxpos:
                    trade = maxpos - pos
                    pos += trade
                    bank_balance -= askPrice[k] * (1+cost) * trade
                if self.prediction[iDay][k] < -self.threshold and pos > -maxpos:
                    trade = pos + maxpos
                    pos -= trade
                    bank_balance += bidPrice[k] * (1-cost) * trade
                volume += trade
                position.append(pos)
                pnl.append(bank_balance + pos*(askPrice[k] + bidPrice[k])/2)
            self.OutOfSampleStats.loc[contract] = [pnl[-1] - bank_balance0, volume, OSR2, OSR2ADJ]
            self.pnl.extend(pnl)
            self.position.extend(position)
        pass
    
    def setThreshold(self, threshold):
        self.threshold = threshold
        pass
    
    def summary(self):
        # sharpe ratio
        # r2
        # max drawdown
        self.summary = pd.concat([self.OutOfSampleStats, self.InSampleStats], axis = 1).fillna(0)
        
        pass