# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:15:15 2018

@author: Marvos
"""

import numpy as np  
from pyfilterbank import FractionalOctaveFilterbank 
#from scipy.io import wavfile
from scipy import stats


fs = 44100
medp= np.load('C:/Users/Marvos/Desktop/Sala_TCC/Pos1/Spyder/medisom.npz')
hpy = medp['h']
Hpy = medp['H']
raw_signal = hpy
init = 0
end = -10
factor = 6
EDT = np.zeros(32)

octfilter = FractionalOctaveFilterbank(sample_rate=44100, order=4, nth_oct=3.0, norm_freq=1000, start_band=-19, end_band=13, edge_correction_percent=0.01, filterfun='py')

hfilt = octfilter.filter(raw_signal)
filtered_signal = hfilt[0]


for band in range(32):
        # Filtering signal
    abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
    sch = np.cumsum(abs_signal[::-1, band]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample: end_sample + 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    EDT[band] = factor*((end_sample/44100))
    