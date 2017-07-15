#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:15:08 2017

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
from signals import Signals
import talib

class talibSignals(Signals):
    def __init__(self, talibSignalList = [], talibSignalParamList = {}):
        Signals.__init__(self)
        self.talibSignalList = talibSignalList
        self.talibSignalParamList = talibSignalParamList
        
    def calcRHS(self, data, sigTable):
        inputarray = data[['open', 'high', 'low', 'close', 'volume']]
        if len(self.talibSignalList) > 0:
            for signal in self.talibSignalList:
                sig = talib.abstract.Function(signal)
                if self.talibSignalParamList and self.talibSignalParamList['signal']:
                    temp = sig(inputarray, self.talibSignalParamList['signal'])
                else:
                    temp = sig(inputarray)
                if type(temp) == pd.Series:
                    temp = pd.DataFrame(temp, columns = [signal], index = data.index)
                elif type(temp) == pd.DataFrame: 
                    temp = temp.set_index(data.index)
                sigTable = pd.concat([sigTable, temp], axis=1)
        return sigTable
