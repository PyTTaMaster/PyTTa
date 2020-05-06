#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:27:15 2019

@author: mtslazarin
"""

import pytta


if __name__ == '__main__':

    device = pytta.get_device_from_user()

    measurementParams = {
        'lengthDomain': 'time',
        'timeLength': 3,
        'samplingRate': 48000,
        'freqMin': 20,
        'freqMax': 20000,
        'device': device,
        'inChannels': [1, 2],
        'comment': 'Testing; 1, 2.'
    }

    msr = pytta.generate.measurement('rec',
                                     **measurementParams)

    med1 = msr.run()
    med1.play()
