#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:40:11 2017

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

class VOI(Signals):
    def __init__(self, signalName = 'VOI', signalParams = []):
        Signals.__init__(self)
        self.signalName = signalName
        self.signalParams = signalParams
        
    def calcRHS(self, data, sigTable):
        diff_Bidprice = np.diff(data['BidPrice1'].values)
        diff_Bidprice = np.insert(diff_Bidprice,0,0,axis = 0)
        diff_Askprice = np.diff(data['AskPrice1'].values)
        diff_Askprice = np.insert(diff_Askprice,0,0,axis = 0)
        
        dVB = ((data['BidVolume1'].values - \
                     data['BidVolume1'].shift(1).fillna(0).values \
                     * (diff_Bidprice == 0) * 1.0) \
                     * (diff_Bidprice >= 0) * 1.0)
        
        dVA = ((data['AskVolume1'].values - \
                     data['AskVolume1'].shift(1).fillna(0).values \
                     * (diff_Askprice == 0) * 1.0) \
                     * (diff_Askprice <= 0) * 1.0)
    
        VOIdata = dVB - dVA            
        sigTable['VOI'] = VOIdata
        return sigTable
    
