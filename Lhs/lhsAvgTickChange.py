#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:57:53 2017

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
from lhs import lhs

class getAvgTickChange(lhs):
    def __init__(self, N, pricecol = 'MidPrice'):
        lhs.__init__(self)
        self.N = N
        self.pricecol = pricecol
    
    def calcLHS(self, data, sigTable):
        sigTable['lhs'] = data[self.pricecol].rolling(self.N).mean().shift(-self.N) - data[self.pricecol]
        return sigTable
        