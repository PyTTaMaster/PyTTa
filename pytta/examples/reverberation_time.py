#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:39:48 2019

@author: joaovitor
"""

import pytta
import os


if __name__ == "__main__":
    path = 'data' + os.sep
    name = 'myir'
    wav = '.wav'
    hdf5 = '.hdf5'
    
    myIRsignal = pytta.read_wav(path + name + wav)
    
    # myIRsignal = pytta.load(path + name + hdf5)
    
    myRT = pytta.rooms.RT(20, myIRsignal, 3)
    # myD50 = pytta.rooms.D(50, myIRsignal, 1)
    # myC80 = pytta.rooms.C(80, myIRsignal, 6)
