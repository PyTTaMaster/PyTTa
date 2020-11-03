#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:15:02 2019

@author: mtslazarin
"""

import pytta


if __name__ == '__main__':

    device = pytta.get_device_from_user()

    sweep = pytta.generate.sweep(fftDegree=18)  # Generate sine sweep SignalObj

    measurementParams = {
        'excitation': sweep,  # Passes the sweep signal as excitation signal
        'samplingRate': 44100,  # Frequency of sampling
        'freqMin': 100,  # Minimum frequency of interest NOT WORKING
        'freqMax': 20000,  # Maximum frequency of interest NOT WORKING
        'device': device,  # Device number provided at runtime
        'inChannels': [1],  # List of hardware channels to be used
        'outChannels': [1],  # List of hardware channels to be used
        'comment': 'Testing; 1, 2.'  # just a comentary
    }

    ms = pytta.generate.measurement('frf',  # Generates the configuration for an impulse response measurement
                                    **measurementParams)

    #%% Run the measurement and process data, return an ImpulseResponse object
    med1 = ms.run()

    #%% Shows the data
    med1.plot_time()
    med1.plot_freq()

    #%% Save the measurement
    path = 'data\\'
    name = 'myir'
    filename = f'{path}{name}'

    # Save PyTTaObj as HDF5 file
    pytta.save(filename, med1)

    # Export wave file of SignalObj
    # pytta.write_wav(filename+'.wav', med1.IR)
