#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:10:28 2017

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

class MPB(Signals):
    def __init__(self, signalName = 'MPB', signalParams = []):
        Signals.__init__(self)
        self.signalName = signalName
        self.signalParams = signalParams
        
    def calcRHS(self, data, sigTable):
        sigTable[self.signalName] = data.AvePrice - data.MidPrice.rolling(2).mean().fillna(data.MidPrice.values[0])
        return sigTable
    
