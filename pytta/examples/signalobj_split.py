#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:00:47 2020

@author: mtslazarin
"""

import pytta

# generates a SignalObj

mySigObj1 = pytta.generate.sin(Arms=0.1)
mySigObj2 = pytta.generate.sin(Arms=0.2)
mySigObj3 = pytta.generate.sin(Arms=0.3)

mySigObjs = pytta.merge(mySigObj1, mySigObj2, mySigObj3)

#%% 
splitedSigObj = mySigObjs.split(channels=[3,2,1])

plitedSigObj2 = pytta.split(mySigObjs, mySigObj1, mySigObj2)