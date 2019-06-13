#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:27:15 2019

@author: mtslazarin
"""

import pytta

msr = pytta.generate.measurement('rec',
                                 lengthDomain='time',
                                 timeLength=3,
                                 samplingRate=48000,
                                 freqMin=20,
                                 freqMax=20000,
                                 device=4,
                                 inChannel=[1],
                                 comment='Testing; 1, 2.')

med1 = msr.run()
