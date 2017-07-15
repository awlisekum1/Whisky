#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:47:48 2017

@author: TaoLuo
"""

import pandas as pd
import numpy as np
import pickle
import copy
import datetime
import os
import matplotlib.pyplot as plt
import lhsTickReturn as tr
import lhsAvgTickChange as atc

class lhsGenerator:
    def __init__(self, param):
        self.param = param
        self.parseParam()
        
    def parseParam(self):
        self.lhsName = self.param['lhsName']
        self.N = self.param['lhsParams']
        self.priceCol = self.param['lhsPrice']
        pass
        
    def generateLhs(self):
        if self.lhsName == 'nTickReturn':
            return self.generateNTickReturn
        if self.lhsName == 'avgTickChange':
            return self.generategetAvgTickChange
    
    def generateNTickReturn(self, data, sigTable):
        lhs = tr.getNtickRet(self.N, self.priceCol)
        return lhs.calcLHS(data, sigTable)
    
    def generategetAvgTickChange(self, data, sigTable):
        lhs = atc.getAvgTickChange(self.N, self.priceCol)
        return lhs.calcLHS(data, sigTable)
