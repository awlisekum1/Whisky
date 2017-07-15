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
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Driver')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Data')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Signals/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Lhs/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Fitter/')
sys.path.append('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Backtest/')
import time
import copy
import json
import numpy as np
import pandas as pd
import driver as dr
import signalGenerator as signalG
import lhsGenerator as lhsG
import backtestGenerator as backtestG
import fitterGenerator as fitterG
import random
import pickle
import talib
import imp
import matplotlib.pyplot as plt
import dataProcessing as dp
imp.reload(backtestG)
imp.reload(fitterG)
imp.reload(lhsG)
imp.reload(signalG)
imp.reload(dr)
imp.reload(dp)

#%%

import time
def tic():
    globals()['tt'] = time.clock()
 
def toc():
    print('\nElapsed time: %.8f seconds\n' % (time.clock()-globals()['tt']))
    
def putJSON(data, filename):
	try:
		jsondata = json.dumps(data, indent=4, skipkeys=True, sort_keys=True)
		fd = open(filename, 'w')
		fd.write(jsondata)
		fd.close()
	except:
		print('ERROR writing', filename)
		pass

def getJSON(filename):
	returndata = {}
	try:
		fd = open(filename, 'r')
		text = fd.read()
		fd.close()
		returndata = json.loads(text)
		# Hm.  this returns unicode keys...
		#returndata = simplejson.loads(text)
	except: 
		print('COULD NOT LOAD:', filename)
	return returndata 

cfgData =      {'DataFilePath':
                '/Users/TaoLuo/Desktop/backtest/Python/HFT Framework',
#                '/Users/TaoLuo/Desktop/backtest/High Frequency Data/temp/DCE/m/day/2016/08/',
                'loadMethod': 'pickle',
                'leadLagFiles': # product:path
                {'rb':'/Users/TaoLuo/Desktop/backtest/High Frequency Data/temp/SHFE/rb/day/2016/08/'
                 },
                'trainRatio': 0.7, 
                'dt': 600, 
                'pricepertick': 10.0,
                'shuffle': False
                }
cfgFitter =    {'fitterName': 'OLS', 
                'fitterParams': None}
cfgSignal =    {'signalNameList': ['OIR', 'MPB','VOI','TALibSignals', 'Lag','macdsignal'], # 'OIR', 'TALibSignals', 'MPB','VOI','Lag'
                'talibSignalList': ['macd','WILLR','RSI','EMA', 'STOCH'],  # 'WILLR','RSI','EMA', 'STOCH'
                'adjSpread': True,
                'signalParams': None,
                'leadLag':['rb'],
                'lags': 10
                }
cfgLhs =       {'lhsName': 'avgTickChange',
                'lhsParams': 6,
                'lhsPrice': 'MidPrice'
                }
cfgBacktester= {'backtesterName': 'simpleBacktester',
                'weight': [0.01,0.99],
                'sample': 'all',
                'fitDayorSession': 'day',
                'cost': 0.000025,
                'threshold': 0.61
                }
cfgDriver =    {'lhsParamRange': [4,15],
                'thresholdRange': [0.4,0.4],
                'crossoverRate': 0.99,
                'mutationRate': 0.99,
                'populationSize': 2,
                'evolveTime': 2,
                'cutRatio': 0.5
                }
Config = {'Driver': cfgDriver,'Data': cfgData, 'Fitter': cfgFitter, 'Signal':cfgSignal, 'Lhs': cfgLhs, 'Backtester': cfgBacktester}

putJSON(Config, 'Config.txt')
config = getJSON('Config.txt')

#%%
tic()
driver = dr.Driver('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Config.txt')
toc()
#driver.setNewConfig('/Users/TaoLuo/Desktop/backtest/Python/HFT Framework/Config.txt')
tic()
driver.backtesting()
#driver.optimizeBacktesting()
#plt.figure()
#plt.plot(np.cumsum(driver.backtester.BTester.OutOfSampleStats.PnL.values))
toc()