#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytta


if __name__ == "__main__":

#    mySignal = pytta.generate.random_noise()  # Generates a SignalObj with a "white noise" random array of data
    mySignal = pytta.generate.sweep()  # Generates a SignalObj with a sine wave with exponential frequency sweeo

    # SignalObj data visualization
    mySignal.plot_time()
    mySignal.plot_freq()

    octFiltParams = {
        'order': 2,  # Second order SOS butterworth filter design
        'nthOct': 3,  # 1/nthOct bands of frequency octaves, between minFreq and maxFreq
        'samplingRate': 44100,  # Frequency of sampling
        'minFreq': 2e1,  # Minimum frequency of filter coverage
        'maxFreq': 2e4,  # Maximum frequency of filter coverage
        'refFreq': 1e3,  # Reference central frequency for octave band divitions
        'base': 10,  # Calculate bandcut with base 10 values: 10**(0.3/nthOct)
    }

    myFilt = pytta.generate.filter('octave', **octFiltParams)  # OctFilter object

    myListFSignal = myFilt.filter(mySignal)  # Apply filters to signal, gets list of filtered signals per channel

    # Filter visualization
    myListFSignal[0].plot_time()
    myListFSignal[0].plot_freq()
