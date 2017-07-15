#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:37:05 2017

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
import talib
import sigOIR 
import sigVOI
import sigTAlibSignals as sigTAlibS
import sigMPB 

import time
def tic():
    globals()['tt'] = time.clock()
 
def toc():
    print('\nElapsed time: %.8f seconds\n' % (time.clock()-globals()['tt']))


class signalGenerator:
    def __init__(self, param):
        self.__param = param
        self._parseParam()
        
    def _parseParam(self):
        self.signalNameList = self.__param['signalNameList']
        self.talibSignalList = self.__param['talibSignalList']
        self.signalParams = self.__param['signalParams']
        self.lagList = self.signalNameList + self.talibSignalList
        self.lags = self.__param['lags']
        self.adjSpread = self.__param['adjSpread']
        self.leadLag = self.__param['leadLag']
        
    def generateSignals(self):
        return self.aggregateSignals
        
    def aggregateSignals(self, data):
        sigTable = pd.DataFrame(None, index = data.index)
        if 'OIR' in self.signalNameList:
            sigTable = self.generateOIR(data,sigTable)
        if 'VOI' in self.signalNameList:
            sigTable = self.generateVOI(data,sigTable)
        if 'MPB' in self.signalNameList:
            sigTable = self.generateMPB(data,sigTable)
        if 'TALibSignals' in self.signalNameList:
            sigTable = self.generateTALibSignals(data, sigTable)
        if 'Lag' in self.signalNameList:
            sigTable = self.generateLaggedSignals(data,sigTable)
        if 'LeadLag' in self.signalNameList:
            sigTable = self.generateLeadLagSignals(data, sigTable, self.leadLag)
        if self.adjSpread:
            sigTable = sigTable.div(data.Spread, axis = 0)

        return sigTable
        
    def generateOIR(self, data, sigTable):
        signal = sigOIR.OIR('OIR', self.signalParams)
        return signal.calcRHS(data, sigTable)
        
    def generateVOI(self, data, sigTable):
        signal = sigVOI.VOI('VOI', self.signalParams)
        return signal.calcRHS(data, sigTable)
        
    def generateMPB(self, data, sigTable):
        signal = sigMPB.MPB('MPB', self.signalParams)
        return signal.calcRHS(data, sigTable)

    def generateTALibSignals(self, data, sigTable):
        signal = sigTAlibS.talibSignals(self.talibSignalList, self.signalParams)
        return signal.calcRHS(data, sigTable)
    
    def generateLaggedSignals(self, data, sigTable):
        sigTable = sigTable.copy()
        for i, iSig in enumerate(self.lagList):
            if iSig in sigTable.columns:
                if type(self.lags) == list:
                    iLag = self.lags[i]
                if type(self.lags) == int:
                    iLag = self.lags
                for l in range(iLag):
                    sigTable.loc[:, '%s_lag%d' % (iSig, l+1)] = sigTable[iSig].shift(l+1)
        return sigTable
        
    def generateLeadLagSignals(self, data, sigTable, leadLag):
        pass
    
