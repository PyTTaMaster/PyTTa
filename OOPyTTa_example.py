#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Oriented Python to Technical Acoustics example file

Created on Fri Aug 31 11:56:54 2018
Last modified on Sun Oct 28 09:56:32 2018

@author: Matheus Lazarin Albert, João Vitor Gutkoski Paes

This file shows a few ways to use the PyTTa package and it''s modules
Consider running this file section by section, without the runfile() method

"""
#%%
import pytta # importing Object Oriented Python in Technical Acoustics

#%% Default properties and device selecting (not to be run inside a script)

## To run this example file using runfile(), consider commenting this section.
#pytta.properties.default # show all the defaul properties and its values
#pytta.properties.list_devices() # list available audio in/out devices and identify the default ones
#
## change the device and sampling rate values within the "default" dict
#pytta.properties.set_default(device = [1,2], samplingRate=51200) 

#%% generate excitation signal
x1 = pytta.generate.sweep() # with default settings
x2 = pytta.generate.sweep(50,16000) # chosing bandwidth
x3 = pytta.generate.sweep(fftDeg = 16,
                          startmargin=0.1,
                          stopmargin=0.9) # chosing some parameters, without ordering
x4 = pytta.generate.sweep(100,10000,48000,17,0.5,0.5) # choosing all parameters, ordered input

#%% creating the frfmeasure object
x = x1 # define signalObj for the measurement example
recTime = pytta.generate.measurement('rec','time',10)
recDeg = pytta.generate.measurement('rec','samples',17)
m1 = pytta.generate.measurement('frf',x) #initializing with selected signalobj
m2 = pytta.generate.measurement('frf') #initializing with default settings, empty excitation
m2.excitation = x #declaring excitation origin, or
# m2.excitation.t = x.t[:] # declaring excitation value
#m3 = pytta.FRFMeasure(x,
#                      device = 'audiobox',
#                      outch = [3,4],
#                      inch = [1,2] ) # initializing with different properties

#%% running the measurement
m = m1 # define measurement object for run example
y = m.run()

#%% plot stuff
y.plot_freq()
y.plot_time()