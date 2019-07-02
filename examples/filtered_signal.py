#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytta


if __name__ == "__main__":

#    mySignal = pytta.generate.noise()
    mySignal = pytta.generate.sweep()

    mySignal.plot_time()
    mySignal.plot_freq()

    myFilt = pytta.generate.filter(order=2,
                                   nthOct=3,
                                   samplingRate=44100,
                                   minFreq=20,
                                   maxFreq=20000,
                                   refFreq=1000,
                                   base=10)

    myFSignal = myFilt.filter(mySignal)

    myFSignal.plot_time()
    myFSignal.plot_freq()
