#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:39:48 2019

@author: joaovitor
"""

import pytta
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


f = loadmat('RIS/RI_mono_1.mat')


mySignal = pytta.SignalObj(f['MonoRIS1_time'], 'time', 44100)


myRT = pytta.rooms.RT(20, mySignal, 3)
myD50 = pytta.rooms.D(50, mySignal, 1)
myC80 = pytta.rooms.C(80, mySignal, 6)
