#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:39:59 2017

@author: TaoLuo
"""

## Get Data

import os
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
import signalGenerator as signalG
import lhsGenerator as lhsG
import backtestGenerator as backtestG
import fitterGenerator as fitterG
import random
import pickle
import talib
import imp
import matplotlib.pyplot as plt

class DataProc:
    def __init__(self, param):
        self.__param = param
        self._parseParam()
        self.CFFEX_TRADING_SESSION = [['09:15:00.000','11:30:00.000'],['13:00:00.000','15:15:00.000']]
        self.SHFE_DAY_TRADING_SESSION = [['09:00:00.000','10:15:00.000'],['10:30:00.000','11:30:00.000'],['13:00:00.000','15:00:00.000']]
        self.SHFE_NIGHT_TRADING_SESSION = [['21:00:00.000','25:30:00.000']]
        self.DCE_DAY_TRADING_SESSION = [['09:15:00.000','11:30:00.000'],['13:30:00.000','15:00:00.000']]
        self.DCE_NIGHT_TRADING_SESSION = [['21:00:00.000','25:30:00.000']]
        self.CME_DAY_TRADING_SESSION = [['09:15:00.000','11:30:00.000'],['13:00:00.000','15:15:00.000']]
        self.CME_NIGHT_TRADING_SESSION = [['21:00:00.000','25:30:00.000']]
        if self.loadMethod == 'csv':
            self.loadAsCSV()
        if self.loadMethod == 'pickle':
            self.loadAsPickle()
        
        self.calcMidPrice()
        self.calcSpread()
        self.calcAveragePrice()
        self.calcOHLC()
        self.partitionTwoSubsets()
        pass
    
    def _parseParam(self):
        self.DataFilePath = self.__param['DataFilePath']
        self.trainRatio = self.__param['trainRatio']
        self.dt = self.__param['dt']
        self.loadMethod = self.__param['loadMethod']
        self.pricepertick = self.__param['pricepertick']
        self.shuffle = self.__param['shuffle']
        self.leadLagFiles = self.__param['leadLagFiles']
        pass
    
    def loadAsCSV(self):
        self.DataFiles = os.listdir(self.DataFilePath)
        self.DateList = set()

        for i in range(len(self.DataFiles)):
            self.DateList.add(self.DataFiles[i][-12:-4])
        self.DateList = sorted(self.DateList)
        if not self.DateList[0].isalnum():
            self.DateList = self.DateList[1:]
    
        self.dataRaw = {}
        self.ActiveContractFiles = {}
        Exchanges = ['CFFEX', 'SHFE', 'DCE', 'CME']
        for i in range(len(self.DateList)):
            DataFilesToRead = [string for string in self.DataFiles if self.DateList[i] in string]
            ActiveAccVolume = 0
            ActiveContract = 0  # use the max AccVolume to decide which is active contract
            Contract = None
            for j in range(len(DataFilesToRead)):
                DataRead = pd.read_csv(self.DataFilePath + DataFilesToRead[j])
                if (DataRead['AccVolume'].iloc[-1] > ActiveAccVolume):
                    ActiveAccVolume = DataRead['AccVolume'].iloc[-1]
                    ActiveContract = j
                    Contract = DataRead
            self.ActiveContractFiles[i] = DataFilesToRead[ActiveContract]
            self.dataRaw[self.ActiveContractFiles[i]] = Contract
            self.dataRaw[self.ActiveContractFiles[i]].loc[:,'Turnover'] = self.dataRaw[self.ActiveContractFiles[i]].loc[:,'AccTurnover'].diff(1).fillna(value = self.dataRaw[self.ActiveContractFiles[i]].AccTurnover.iloc[0])

        filenames = str.split(self.DataFilePath, sep = '/')
        self.ExchangeInfo = ['', 'Day']
        for name in Exchanges:
            if name in filenames:
                self.ExchangeInfo[0] = name
            if 'day' in filenames:
                self.ExchangeInfo[1] = 'day'
            elif 'night' in filenames:
                self.ExchangeInfo[1] = 'night'
                
        self.loadLeadLagData()
        self.cleanData()
        pass
        
    def loadLeadLagData(self):
        ## load leadLag files
        if self.leadLagFiles != {}:
            for product in self.leadLagFiles.keys():
                leadLagFiles = os.listdir(self.leadLagFiles[product])
                DateList = set()
                for i,file in enumerate(leadLagFiles):
                    DateList.add(leadLagFiles[i][-12:-4])
                DateList = sorted(DateList)
                if not DateList[0].isalnum():
                    DateList = DateList[1:]

                for i,date in enumerate(DateList):
                    DataFilesToRead = [string for string in leadLagFiles if DateList[i] in string]
                    ActiveAccVolume = 0
                    ActiveContract = 0  # use the max AccVolume to decide which is active contract
                    Contract = None
                    for j in range(len(DataFilesToRead)):
                        DataRead = pd.read_csv(self.leadLagFiles[product] + DataFilesToRead[j])
                        if (DataRead['AccVolume'].iloc[-1] > ActiveAccVolume):
                            ActiveAccVolume = DataRead['AccVolume'].iloc[-1]
                            ActiveContract = j
                            Contract = DataRead
                    idx = np.searchsorted(self.DateList, date)
                    timeIdx = np.searchsorted(Contract.TimeStamp, self.dataRaw[self.ActiveContractFiles[idx]].TimeStamp) - 1
                    self.dataRaw[self.ActiveContractFiles[idx]].loc[:,product+'MidPrice'] = (Contract.iloc[timeIdx].AskPrice1.values + Contract.iloc[timeIdx].BidPrice1.values)/2
        pass
        
    def cleanData(self):
        self.tradingSession = None
        if self.ExchangeInfo[0] == 'CFFEX':
            self.tradingSession = self.Get_CFFEX_TradingSession()
        elif self.ExchangeInfo[0] == 'SHFE' and self.ExchangeInfo[1] == 'day':
            self.tradingSession = self.Get_SHFE_Day_TradingSession()
        elif self.ExchangeInfo[0] == 'SHFE' and self.ExchangeInfo[1] == 'night':
            self.tradingSession = self.Get_SHFE_Night_TradingSession()
        elif self.ExchangeInfo[0] == 'DCE' and self.ExchangeInfo[1] == 'day':
            self.tradingSession = self.Get_DCE_Day_TradingSession()
        elif self.ExchangeInfo[0] == 'DCE' and self.ExchangeInfo[1] == 'night':
            self.tradingSession = self.Get_DCE_Night_TradingSession()
        elif self.ExchangeInfo[0] == 'CME' and self.ExchangeInfo[1] == 'day':
            self.tradingSession = self.Get_CME_Day_TradingSession()
        elif self.ExchangeInfo[0] == 'CME' and self.ExchangeInfo[1] == 'night':
            self.tradingSession = self.Get_CME_Night_TradingSession()
        self.dataCleaned = {}
        for contract in self.ActiveContractFiles.keys():
            splitdata = {}
            for i in range(len(self.tradingSession)):
                mask1 = (self.dataRaw[self.ActiveContractFiles[contract]].UpdateTime > self.tradingSession[i][0])
                mask2 = (self.dataRaw[self.ActiveContractFiles[contract]].UpdateTime < self.tradingSession[i][1])
                mask3 = (self.dataRaw[self.ActiveContractFiles[contract]].AskPrice1 > 0)
                mask4 = (self.dataRaw[self.ActiveContractFiles[contract]].BidPrice1 > 0)
                mask5 = (self.dataRaw[self.ActiveContractFiles[contract]].AskVolume1 * self.dataRaw[self.ActiveContractFiles[contract]].BidVolume1 > 0)
                mask6 = ((self.dataRaw[self.ActiveContractFiles[contract]].AskPrice1 - self.dataRaw[self.ActiveContractFiles[contract]].BidPrice1)/self.dataRaw[self.ActiveContractFiles[contract]].AskPrice1 < 0.01)
                mask = mask1 & mask2 & mask3 & mask4 & mask5 & mask6
                temp = self.dataRaw[self.ActiveContractFiles[contract]].loc[mask].copy()
                if len(str(int(temp.TimeStamp.values[0]))) < 19:
                    multiple = 10**(19-len(str(int(temp.TimeStamp.values[0]))))
                    temp['TimeStamp'] *= multiple
                temp.loc[:,'UTime'] = pd.to_datetime(temp['TimeStamp']).values
                temp = temp.set_index('UTime')
                if len(temp) > 0:
                    splitdata[i] = copy.deepcopy(temp) 
            self.dataCleaned[self.ActiveContractFiles[contract]] = splitdata
        pass
    
    def calcOHLC(self):
        for contract in self.ActiveContractFiles.keys():
            for i in range(len(self.tradingSession)):
                price = self.dataCleaned[self.ActiveContractFiles[contract]][i].MidPrice
                volume = self.dataCleaned[self.ActiveContractFiles[contract]][i].Volume
                self.dataCleaned[self.ActiveContractFiles[contract]][i] = self.dataCleaned[self.ActiveContractFiles[contract]][i].copy()
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'open'] = price.rolling(self.dt, min_periods=1).apply(lambda x:x[0]).values
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'high'] = price.rolling(self.dt, min_periods=1).max().values
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'low'] = price.rolling(self.dt, min_periods=1).min().values
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'close'] = price.rolling(self.dt, min_periods=1).apply(lambda x:x[-1]).values
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'volume'] = volume.rolling(self.dt, min_periods=1).sum().values
        pass
    
    def calcMidPrice(self):
        for contract in self.ActiveContractFiles.keys():
            for i in range(len(self.tradingSession)):
                self.dataCleaned[self.ActiveContractFiles[contract]][i] = self.dataCleaned[self.ActiveContractFiles[contract]][i].copy()
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'MidPrice'] = \
                    (self.dataCleaned[self.ActiveContractFiles[contract]][i].AskPrice1.values + \
                    self.dataCleaned[self.ActiveContractFiles[contract]][i].BidPrice1.values)/2
        pass
    
    def calcSpread(self):
        for contract in self.ActiveContractFiles.keys():
            for i in range(len(self.tradingSession)):
                self.dataCleaned[self.ActiveContractFiles[contract]][i] = self.dataCleaned[self.ActiveContractFiles[contract]][i].copy()
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'Spread'] = \
                    self.dataCleaned[self.ActiveContractFiles[contract]][i].AskPrice1.values - \
                    self.dataCleaned[self.ActiveContractFiles[contract]][i].BidPrice1.values
        pass
    
    def calcAveragePrice(self):
        for contract in self.ActiveContractFiles.keys():
            for i in range(len(self.tradingSession)):
                self.dataCleaned[self.ActiveContractFiles[contract]][i] = self.dataCleaned[self.ActiveContractFiles[contract]][i].copy()
                self.dataCleaned[self.ActiveContractFiles[contract]][i].loc[:,'AvePrice'] = (self.dataCleaned[self.ActiveContractFiles[contract]][i].Turnover / self.pricepertick / \
                                                (self.dataCleaned[self.ActiveContractFiles[contract]][i].Volume)).fillna(method = 'ffill')  
        pass
    
    def calcSignals(self, SignalGenerator):
        self.dataSignal = {}
        for contract in self.ActiveContractFiles.keys():
            dataSignal = {}
            for i in range(len(self.tradingSession)):
                dataSignal[i] = SignalGenerator(self.dataCleaned[self.ActiveContractFiles[contract]][i])
            self.dataSignal[self.ActiveContractFiles[contract]] = dataSignal
        self.signalList = self.dataSignal[self.ActiveContractFiles[contract]][0].columns
        
        pass        
                
    def calcLhs(self, LhsGenerator):
        for contract in self.ActiveContractFiles.keys():
            for i in range(len(self.tradingSession)):
                self.dataSignal[self.ActiveContractFiles[contract]][i] = LhsGenerator(self.dataCleaned[self.ActiveContractFiles[contract]][i], self.dataSignal[self.ActiveContractFiles[contract]][i])
#        self.dropNA()
        pass
    
#    def dropNA(self):
#        for contract in self.ActiveContractFiles.keys():
#            for i in range(len(self.tradingSession)):
#                self.dataSignal[self.ActiveContractFiles[contract]][i] = self.dataSignal[self.ActiveContractFiles[contract]][i].dropna(how = 'any')
#        pass

    def dumpAsPickle(self, DataFilePath = ""):
        pickleData = {'ActiveContractFiles':self.ActiveContractFiles,
                      'dataCleaned':self.dataCleaned,
                      'dataSignal':self.dataSignal,
                      'tradingSession': self.tradingSession,
                      'ExchangeInfo':self.ExchangeInfo,
                      'DataFiles':self.DataFiles,
                      'DateList':self.DateList,
                      'signalList':self.signalList,
                      'trainSet':self.trainSet,
                      'testSet':self.testSet
                      }
        if DataFilePath == "":
            pickle.dump(pickleData, open(self.DataFilePath + '/dataCleaned.p','wb'))
        else:
            pickle.dump(pickleData, open(DataFilePath + '/dataCleaned.p','wb'))
        pass
    
    def loadAsPickle(self, DataFilePath = ""):
        if DataFilePath == "":
            pickleData = pickle.load(open(self.DataFilePath + '/dataCleaned.p','rb'))
        else:
            pickleData = pickle.load(open(DataFilePath + '/dataCleaned.p','rb'))
        self.dataCleaned = pickleData['dataCleaned']
        self.ActiveContractFiles = pickleData['ActiveContractFiles']
        self.dataSignal = pickleData['dataSignal']
        self.tradingSession = pickleData['tradingSession']
        self.ExchangeInfo = pickleData['ExchangeInfo']
        self.DataFiles = pickleData['DataFiles']
        self.DateList = pickleData['DateList']
        self.signalList = pickleData['signalList']
        self.trainSet = pickleData['trainSet']
        self.testSet = pickleData['testSet']
        pass
        
    def partitionTwoSubsets(self):
        templist = list(self.ActiveContractFiles.values())
        if self.shuffle == True:
            random.shuffle(templist)
        trainN = (int)(len(templist) * self.trainRatio)
        self.trainSet = templist[:trainN]
        self.testSet = templist[trainN:]
        pass    
    
    def Get_CFFEX_TradingSession(self):
        return self.CFFEX_TRADING_SESSION
    def Get_SHFE_Day_TradingSession(self):
        return self.SHFE_DAY_TRADING_SESSION
    def Get_SHFE_Night_TradingSession(self):
        return self.SHFE_NIGHT_TRADING_SESSION   
    def Get_DCE_Day_TradingSession(self):
        return self.DCE_DAY_TRADING_SESSION     
    def Get_DCE_Night_TradingSession(self):
        return self.DCE_NIGHT_TRADING_SESSION
    def Get_CME_Day_TradingSession(self):
        return self.CME_DAY_TRADING_SESSION
    def Get_CME_Night_TradingSession(self):
        return self.CME_NIGHT_TRADING_SESSION
        

        

