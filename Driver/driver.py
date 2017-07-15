#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:25:56 2017

@author: TaoLuo
"""

import os
import sys
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Data')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Signals/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Lhs/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Fitter/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Backtest/')
import copy
#from multiprocessing import Pool 
import multiprocessing
import time
import json
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
import dataProcessing as dataP
imp.reload(backtestG)
imp.reload(fitterG)
imp.reload(lhsG)
imp.reload(signalG)
imp.reload(dataP)


def start_process():
    print('Starting',multiprocessing.current_process().name)


def tic():
    globals()['tt'] = time.clock()
 
def toc():
    print('\nElapsed time: %.8f seconds\n' % (time.clock()-globals()['tt']))
  

class Driver():
    def __init__(self, cfgName):
        self.__param = self._getJSON(cfgName)
        self._parseParam()
        self._getData()
        self.__setNewParam = False
        pass

    def _parseParam(self):
        self.dataParam = self.__param['Data']
        self.fitterParam = self.__param['Fitter']
        self.signalParam = self.__param['Signal']
        self.lhsParam = self.__param['Lhs']
        self.backtestParam = self.__param['Backtester']   
        self.driverParam =   self.__param['Driver']

        self.lhsParamRange = self.driverParam['lhsParamRange']
        self.thresholdRange = self.driverParam['thresholdRange']
        self.crossoverRate = self.driverParam['crossoverRate']
        self.mutationRate = self.driverParam['mutationRate']
        self.populationSize = self.driverParam['populationSize']
        self.evolveTime = self.driverParam['evolveTime']
        self.cutRatio = self.driverParam['cutRatio']
        pass
    
    def setNewConfig(self, cfgName):
        newParam = self._getJSON(cfgName)
        self.__setNewParam = True
        if self.dataParam != newParam['Data']:
            self.dataParam = newParam['Data']
            self._getData()
        if self.fitterParam != newParam['Fitter']:
            self.fitterParam = newParam['Fitter']
            self._getFitter()
        if self.signalParam != newParam['Signal']:
            self.signalParam = newParam['Signal']
            self._getSignalList()
            self.lhsParam = newParam['Lhs']
            self._getLhs()
        if self.lhsParam != newParam['Lhs']:
            self.lhsParam = newParam['Lhs']
            self._getLhs()
        if self.backtestParam != newParam['Backtester']:
            self.backtestParam = newParam['Backtester']
            self._getBacktester()
        pass
        

    def _getData(self):
        self.data = dataP.DataProc(self.dataParam)
        pass
        
    def _getSignalList(self):
        self.signal = signalG.signalGenerator(self.signalParam)
        self.signalGenerator = self.signal.generateSignals()
        self.data.calcSignals(self.signalGenerator)
        pass
    
    def _getLhs(self):
        self.lhs = lhsG.lhsGenerator(self.lhsParam)
        self.lhsGenerator = self.lhs.generateLhs()
        self.data.calcLhs(self.lhsGenerator)
        pass
    
    def _getFitter(self):
        self.fitter = fitterG.fitterGenerator(self.fitterParam)
        self.fitterGenerator = self.fitter.generateFitter()
        pass
    
    def _getBacktester(self):
        self.backtester = backtestG.backtestGenerator(self.backtestParam)
        self.backtestGenerator = self.backtester.generateBacktester()
        pass
    
    def optimizeBacktesting(self):
        # cutRatio, sort_values(Return) should be put into params later
        self._getSignalList()
        self._getLhs()
        self._getFitter()
        self._getBacktester()
        self.backtestGenerator(self.data, self.fitterGenerator, signals = self.data.signalList)
        
        ## try different combinations of signals and params(threshold/lhsParam)
        ## goal: max pnl/sharpe ratio/win ratio, min drawdown
        
        self.backtestParam['sample'] = 'train'        
        self.GeneticAlgo()    

#        for i in range(nTimes):
#            self.lhsParam['lhsParams'] = random.randint(4,15)
#            w1 = random.random()
#            self.backtestParam['weight'] = [w1, 1-w1]
#            self.backtestParam['threshold'] = random.random()*0.4 + 0.4
#            self._getLhs()
#            self._getBacktester()
#            sigList = np.random.randint(0,2,size = (len(self.data.signalList),))
#            signalList = [self.data.signalList[i] for i,num in enumerate(sigList) if num == 1]
#            self.backtestGenerator(self.data, self.fitterGenerator, signalList)
#            self.initResultTable.loc[i] = self.backtester.BTester.statsTable.loc['Stats'].tolist() + list(self.backtestParam['weight']) + [self.backtestParam['threshold'], self.lhsParam['lhsParams']] + list(sigList)
#            print('!Current Training Loop: ', i)
         
        cut = int(self.initResultTable.shape[0] * self.cutRatio)
        cols = ['test'+ colName for colName in self.backtester.BTester.statsTable.columns]
        testStats = pd.DataFrame(None, columns = cols)
        self.backtestParam['sample'] = 'test'
        self._getBacktester()
        self.testResultTable = self.initResultTable.sort_values('Return').iloc[-cut:]
        for i in self.testResultTable.index:
            self.lhsParam['lhsParams'] = int(self.testResultTable.loc[i].lhsParam)
            self.backtestParam['weight'] = self.testResultTable.loc[i][['preWt', 'aftWt']].tolist()
            self.backtestParam['threshold'] = self.testResultTable.loc[i].Threshold
            self._getLhs()
            self._getBacktester()
            sigList = self.testResultTable.loc[i][10:].values.astype(int)
            signalList = [self.data.signalList[i] for i,num in enumerate(sigList) if num == 1]
            self.backtestGenerator(self.data, self.fitterGenerator, signalList)
            testStats.loc[i] = self.backtester.BTester.statsTable.loc['Stats'].tolist()
            print('!Current Testing Loop: ', i)
        self.testResultTable = pd.concat([testStats, self.testResultTable], axis = 1)
        
        
        self.backtestParam['sample'] = 'all'
        self.testResultTable = self.testResultTable.sort_values('testReturn')
        self.lhsParam['lhsParams'] = int(self.testResultTable.iloc[-1].lhsParam)
        self.backtestParam['weight'] = self.testResultTable.iloc[-1][['preWt', 'aftWt']].tolist()
        self.backtestParam['threshold'] = self.testResultTable.iloc[-1].Threshold
        self._getLhs()
        self._getBacktester()
        sigList = self.testResultTable.iloc[-1][16:].values.astype(int)
        signalList = [self.data.signalList[i] for i,num in enumerate(sigList) if num == 1]
        self.backtestGenerator(self.data, self.fitterGenerator, signalList)
        
        ## save results to csv files
        pass
       
    
    def backtesting(self):     
        if self.__setNewParam == False:
            self._getSignalList()
            self._getLhs()
            self._getFitter()
            self._getBacktester()
        self.backtestGenerator(self.data, self.fitterGenerator, signals = self.data.signalList)
        plt.plot(self.backtester.BTester.pnl)
        pass
    

    def _putJSON(self, data, filename):
        try:
            jsondata = json.dumps(data, indent=4, skipkeys=True, sort_keys=True)
            fd = open(filename, 'w')
            fd.write(jsondata)
            fd.close()
        except:
             print('ERROR writing', filename)
        pass

    def _getJSON(self,filename):
        returndata = {}
        try:
            fd = open(filename, 'r')
            text = fd.read()
            fd.close()
            returndata = json.loads(text)
        except: 
            print('COULD NOT LOAD:', filename)
        return returndata 
        
    def paramBacktest(self, params):
        self.lhsParam['lhsParams'] = params[0]
        self.backtestParam['weight'] = params[1:3]
        self.backtestParam['threshold'] = params[3]
        self._getLhs()
        self._getBacktester()
        sigList = np.array(params[4:]).astype(int)
        signalList = [self.data.signalList[i] for i,num in enumerate(sigList) if num == 1]
        self.backtestGenerator(self.data, self.fitterGenerator, signalList)
        result = self.backtester.BTester.statsTable.loc['Stats'].tolist()
        return result    
        
    def GeneticAlgo(self):
        M = self.populationSize
        N = self.evolveTime
        chromo = self.GARandomGene(M)
        for i in range(N):
            print("GeneticAlgo: ",i)
            chromo_mutation = self.GAMutation(chromo)
            chromo_crossover = self.GACrossOver(chromo)
            chromo_randomGene = self.GARandomGene(M)
            chromo.extend(chromo_mutation)
            chromo.extend(chromo_crossover)
            chromo.extend(chromo_randomGene)
            
            row = len(chromo)
            results = []

#            pool = multiprocessing.Pool(processes = 4)
            for j in range(row):
                result = self.paramBacktest(chromo[j])
#                result = pool.apply_async(self.paramBacktest, chromo[j])
                results.append(result)
#            pool.close()
#            pool.join()
            
            results = pd.DataFrame(results, columns = self.backtester.BTester.statsTable.columns)
            results = results.sort_values('Return')
            results = results.iloc[-M:]
            resultIdx = results.index.values.astype(int)

            chromo_filted = []
            for j in resultIdx:
                chromo_filted.append(chromo[j])
            chromo = chromo_filted
        
        colParams = ['lhsParam', 'preWt','aftWt', 'Threshold'] + list(self.data.signalList)
        bestISStats = results
        bestISParams = pd.DataFrame(chromo, columns = colParams, index = results.index) 
        self.initResultTable = pd.concat([bestISStats, bestISParams], axis = 1)
        pass
            
            
    def GARandomGene(self,size):
        chromo = []
        for i in range(size):
            temp_chromo = []
            lhsParams = random.randint(self.lhsParamRange[0],self.lhsParamRange[1])
            preWt = random.random()
            aftWt = 1 - preWt
            threshold = self.thresholdRange[0] + random.random()*self.thresholdRange[1]
            temp_chromo.extend([lhsParams, preWt, aftWt, threshold])
            sigList = np.random.randint(0,2,size = (len(self.data.signalList),))
            temp_chromo.extend(sigList)
            chromo.append(temp_chromo)
        return chromo
        
    def GACrossOver(self, chromo):
        row, col = len(chromo), len(chromo[0])
        chromo_crossover = []
        for i in range(0, row, 2):
            chromo1 = chromo[i][:]
            chromo2 = chromo[i+1][:]
            if random.random() < self.crossoverRate:
                pos = random.randint(0,col-1)
                chromo1[pos:], chromo2[pos:] = chromo2[pos:], chromo1[pos:]
            chromo_crossover.append(chromo1)
            chromo_crossover.append(chromo2)
        return chromo_crossover

    def GAMutation(self, chromo):
        row, col = len(chromo), len(chromo[0])
        chromo_mutation = []
        for i in range(row):
            chromo_i = chromo[i][:]
            if random.random() < self.mutationRate:
                lhsParams = random.randint(self.lhsParamRange[0],self.lhsParamRange[1])
                preWt = random.random()
                aftWt = 1 - preWt
                threshold = self.thresholdRange[0] + random.random()*self.thresholdRange[1]
                chromo_i[:4] = [lhsParams, preWt, aftWt, threshold]
            chromo_mutation.append(chromo_i)
        return chromo_mutation
        

