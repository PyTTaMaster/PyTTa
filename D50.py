# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:36:11 2018

@author: Marvos
"""


fs = 44100
medp= np.load('C:/Users/Marvos/Desktop/Sala_TCC/Pos1/Spyder/medisom.npz')
hpy = medp['h']
Hpy = medp['H']
raw_signal = hpy

octfilter = FractionalOctaveFilterbank(sample_rate=44100, order=4, nth_oct=3.0, norm_freq=1000, start_band=-19, end_band=13, edge_correction_percent=0.01, filterfun='py')

hfilt = octfilter.filter(raw_signal)
filtered_signal = hfilt[0]
h = filtered_signal**2
fff = hfilt[1]

f_filtro = np.fromiter(fff.keys(), dtype=float)
f_filtro = np.transpose(f_filtro)

d_50 = np.zeros(f_filtro.size)

for band in range(33):
       
        h = filtered_signal**2
        t = int((50/1000.0)*fs + 1)
        d_50[band] = 10.0*np.log10((np.sum(h[:t])/np.sum(h[:])))