#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:35:49 2017

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

class OIR(Signals):
    def __init__(self, signalName = 'OIR', signalParams = []):
        Signals.__init__(self)
        self.signalName = signalName
        self.signalParams = signalParams
        
    def calcRHS(self, data, sigTable):
        sigTable[self.signalName] = (data['BidVolume1'] - data['AskVolume1'])/1.0/(data['BidVolume1'] + data['AskVolume1'])
        return sigTable
    
