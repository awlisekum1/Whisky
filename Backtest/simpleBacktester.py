#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:59:59 2017

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
import time
def tic():
    globals()['tt'] = time.clock()
 
def toc():
    print('\nElapsed time: %.8f seconds\n' % (time.clock()-globals()['tt']))




class simpleBacktester(Backtester):
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
        if self.sample == 'all':
            self.mainContracts = sorted(list(bookData.ActiveContractFiles.values()))
            self.nMainContracts = len(bookData.ActiveContractFiles.keys())
        if self.sample == 'train':
            self.mainContracts = bookData.trainSet
            self.nMainContracts = len(bookData.trainSet)
        if self.sample == 'test':
            self.mainContracts = bookData.testSet
            self.nMainContracts = len(bookData.testSet)
        self.signals = signals
        
        self.fitting(bookData, fitter, signals)
        self.predicting(bookData, signals)
        self.orderLogic(bookData, self.cost)
        self.summary()
        pass
    
    def fitting(self, bookData, fitter, signals):
        self.regressors = {}
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
        

        for contract in range(self.nMainContracts):
            regData = pd.DataFrame(None)
            for i in range(len(bookData.tradingSession)):
                regData = regData.append(bookData.dataSignal[self.mainContracts[contract]][i])
            regData = regData.dropna(how = 'any')
            self.regressors[contract] = fitter(regData['lhs'], regData[signals])
            ISR2 = self.regressors[contract].rsquared
            ISR2ADJ = self.regressors[contract].rsquared_adj
            tstat = self.regressors[contract].tvalues.values
            coefs = self.regressors[contract].params.values
            self.InSampleStats.loc[self.mainContracts[contract]] = np.r_[ISR2, ISR2ADJ, tstat, coefs]
        pass
    
    def predicting(self, bookData, signals):
        # param1:
        #1 rolling prediction
        #2 prediction for out of sample
        self.prediction = {}
        self.predIdx = {}
        self.response = {}
        for iDay in range(self.nWeights, self.nMainContracts):
            predictData = pd.DataFrame(None)
            for i in range(len(bookData.tradingSession)):
                predictData = predictData.append(bookData.dataSignal[self.mainContracts[iDay]][i])
            predictData = predictData.dropna(how = 'any')
            self.prediction[iDay] = np.array([0.0] * len(predictData))
            self.predIdx[iDay] = predictData.index
            self.response[iDay] = predictData['lhs']
            for iw in range(self.nWeights):
                self.prediction[iDay] += self.weights[self.nWeights - iw - 1] * self.regressors[iDay - iw - 1].predict(predictData[signals])  
            
        pass
    
    def orderLogic(self, bookData, cost): 
        self.pnl = [0]
        self.position = [0]
        maxpos = 1     
        
        self.OutOfSampleStats = pd.DataFrame(columns = ['PnL', 'Volume','OSR2','OSR2ADJ'])
        for iDay in self.prediction.keys():
#            print(iDay, self.mainContracts[iDay])
            bank_balance0,  bank_balance = self.pnl[-1], self.pnl[-1]

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
                index = bookData.dataSignal[self.mainContracts[iDay]][i].index
#                orderData = orderData.append(bookData.dataCleaned[self.mainContracts[iDay]][i][['AskPrice1', 'BidPrice1']])
                orderData = orderData.append(bookData.dataCleaned[self.mainContracts[iDay]][i][['AskPrice1', 'BidPrice1']])
            orderData = orderData.loc[self.predIdx[iDay]]
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
            self.OutOfSampleStats.loc[self.mainContracts[iDay]] = [pnl[-1] - bank_balance0, volume, OSR2, OSR2ADJ]
            self.pnl.extend(pnl)
            self.position.extend(position)
        pass
    
    def summary(self):
        # sharpe ratio
        # r2
        # max drawdown
        self.summaryTable = pd.concat([self.OutOfSampleStats, self.InSampleStats], axis = 1).fillna(0)
        
        winRatio = sum(self.OutOfSampleStats.PnL.values > 0) / len(self.OutOfSampleStats.PnL.values)
        ret = sum(self.OutOfSampleStats.PnL.values)
        stdev = np.std(self.OutOfSampleStats.PnL.values)
        sharpe = np.mean(self.OutOfSampleStats.PnL.values) / stdev
        maxDrawdown, cum, curmax = 100000000,0,0
        for i in self.OutOfSampleStats.PnL.values:
            curmax = max(curmax, cum)
            cum += i
            maxDrawdown = min(maxDrawdown, cum-curmax)
        calmarRatio = np.mean(self.OutOfSampleStats.PnL.values) / maxDrawdown
        self.statsTable = pd.DataFrame(None,columns = [['Return','SharpeRatio','Std','WinRatio','MaxDrawdown','CalmarRatio']])
        self.statsTable.loc['Stats'] = [ret, sharpe, stdev, winRatio, maxDrawdown, calmarRatio]

        pass
