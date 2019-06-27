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
                                freqMin=20,
                                freqMax=20000,
                                device=4,
                                inChannel=[1],
                                outChannel=[1, 2],
                                comment='Testing; 1, 2.')

med1 = ms.run()
