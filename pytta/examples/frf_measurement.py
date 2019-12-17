#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:15:02 2019

@author: mtslazarin
"""

import pytta

sweep = pytta.generate.sweep(fftDegree=18)
ms = pytta.generate.measurement('frf',
                                excitation=sweep,
                                samplingRate=44100,
                                freqMin=100,
                                freqMax=20000,
                                device=13,
                                inChannels=[1],
                                outChannels=[1],
                                comment='Testing; 1, 2.')
#%%
med1 = ms.run()

#%%
med1.plot_time()
med1.plot_freq()

#%%
an = pytta.rooms.analyse(med1.IR, 'RT', 20, nthOct=3, minFreq=60, maxFreq=20000,
                        plotLundebyResults=False, IRManualCut=1)

