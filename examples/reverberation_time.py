#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 16:39:48 2019

@authors: joaovitor, mtslazarin
"""

import pytta
import os


if __name__ == "__main__":
    path = 'RIS' + os.sep
    name = 'scene9_RIR_LS1_MP1_Dodecahedron'
    wav = '.wav'
    hdf5 = '.hdf5'
    
    myIRsignal = pytta.read_wav(path + name + wav)
    
    # myIRsignal = pytta.load(path + name + hdf5)
    
    myRT = pytta.rooms.analyse(myIRsignal,
                               'RT', 20,
                               nthOct=3,
                               minFreq=20,
                               maxFreq=10000)
    
    myRT.plot_bars()
