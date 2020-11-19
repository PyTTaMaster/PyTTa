#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:06:21 2019

@author: mtslazarin
"""

import pytta


if __name__ == '__main__':

    device = pytta.get_device_from_user()

    sweep = pytta.generate.sweep(fftDegree=18)

    measurementParams = {
        'excitation': sweep,
        'samplingRate': 44100,
        'freqMin': 20,
        'freqMax': 20000,
        'device': device,
        'inChannels': [1],
        'outChannels': [1, 2],
        'comment': 'Testing; 1, 2.'
    }

    ms = pytta.generate.measurement('playrec',
                                    **measurementParams)

    med1 = ms.run()
    med1.play()
