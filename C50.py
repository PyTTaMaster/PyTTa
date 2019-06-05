# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:20:37 2018

@author: Marvos
"""

import numpy as np  
from pyfilterbank import FractionalOctaveFilterbank 
#from scipy.io import wavfile
from scipy import stats
import matplotlib.pyplot as plt


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

c_50 = np.zeros(f_filtro.size)

for band in range(33):
       
        h = filtered_signal**2
        t = int((50/1000.0)*fs + 1)
        c_50[band] = 10.0*np.log10((np.sum(h[:t])/np.sum(h[t:])))





# =============================================================================
# #f2 = (  100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000)
# #t20_p = t20[9:30]
# plt.figure(figsize=(10, 5))
# p1 = plt.semilogx(f_filtro, t20, 'k--')
# #p2 = plt.semilogx(f2,t20_m, 'b--')
# #plt.legend((p1[0], p2[0]),('Python','Matlab'))
# plt.xscale('log')
# plt.axis([100, 5000, 0, 3])
# plt.xlabel(r'$F$ in Hz')
# plt.ylabel(r'$T20$')
# plt.xticks(f2, ('100', '125', '160', '200', '250', '314', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000', '10000'))
# plt.title('Comparação entre valores de T20 calculados em Python e em Matlab')
# #plt.savefig('T20_PxM.pdf')
# plt.show()
# =============================================================================
