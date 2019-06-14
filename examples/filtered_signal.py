#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytta


if __name__ == "__main__":

    myNoise = pytta.generate.noise()
    mySweep = pytta.generate.sweep()

    myFilt = pytta.generate.filter(order=2,
                                     nthOct=6,
                                     samplingRate=44100,
                                     minFreq=20,
                                     maxFreq=20000,
                                     refFreq=1000,
                                     base=10)

    myFSignal = myFilt.filter(myNoise)

    myFSignal.plot_time()
    myFSignal.plot_freq()
