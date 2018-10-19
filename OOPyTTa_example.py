#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:56:54 2018

@author: mtslazarin
"""

import OOPyTTa as pytta # importing Object Oriented Python in Technical Acoustics
import sounddevice as sd

#%% frfmeasure class arguments

#sd.query_devices()
device = 'built in' # [in device number,out device number] or 'device name' from sounddevices.query_devices()
inch = [1] # [input 1 channel number, input 2 channel number...]
outch = [1,2] # [output 1 channel number, output 2 channel number]
Fs = 44100 # [Hz] Sample rate
Finf = 20 # [Hz] sweep inferior frequency limit
Fsup = 20000 # [Hz] sweep inferior frequency limit
fftdeg = 16 # 2^(FFT degree) [samples] (related to  x(t) signal length)
stopmargin = 1 # [s] record some silence after sweep
comment='Setup teste de Medição' # comment about the measurement setup

#%% creating the excitation signal

x = pytta.generate(44100,20,20000,16,1)
#%% creating the frfmeasure object

m = pytta.frfmeasure(device,inch,outch,Fs,Finf,Fsup,comment,x)

#%% running the measurement

m1 = m.run()

#%% plot stuff

m1.plot_freq()