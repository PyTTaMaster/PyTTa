# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:27:44 2018

@author: Marvos, Chum4k3r
"""

from pytta.properties import default
import numpy as np
from pyfilterbank import FractionalOctaveFilterbank
from scipy import stats
import matplotlib.pyplot as plt

##%% Legacy code
def T20(h,
        Fs = default['samplingRate'],       
        init = -5,
        end = -25,
        factor = 3,
        t20 = np.zeros(33)):
    
            raw_signal = h.timeSignal[:]
            raw_signal = np.array(raw_signal)
            octfilter = FractionalOctaveFilterbank(sample_rate=44100, order=4,
                                                   nth_oct=3.0, norm_freq=1000,
                                                   start_band=-19, end_band=13,
                                                   edge_correction_percent=0.01,
                                                   filterfun='py')
    
            hfilt = octfilter.filter(raw_signal)
            filtered_signal = hfilt[0]
            f_3rd = np.fromiter(hfilt[1].keys(), dtype = float)
            t20_result = [t20, f_3rd]
        
            for band in range(33):
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
                x = np.arange(init_sample, end_sample + 1) / Fs
                y = sch_db[init_sample: end_sample + 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        
            # Reverberation time (T30, T20, T10 or EDT)
                db_regress_init = (init - intercept) / slope
                db_regress_end = (end - intercept) / slope
                t20[band] = 3*((end_sample/44100)-(init_sample/44100))
                t20_result = [t20, f_3rd]
            return t20_result

def plot_T20(t20_result):
    """
    Frequency domain plotting method
    """
    plt.figure(figsize=(10, 5))
    p1 = plt.semilogx(t20_result[1], t20_result[0], 'k--')
    #p2 = plt.semilogx(f2,t20_m, 'b--')
    #plt.legend((p1[0], p2[0]),('Python','Matlab'))
    plt.xscale('log')
    plt.axis([100, 5000, 0, 1.1*np.max(t20_result[0])])
    plt.xlabel(r'$F$ in Hz')
    plt.ylabel(r'$T20$')
    plt.xticks(t20_result[1], ('100', '125', '160', '200', '250', '314', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000', '10000'))
    plt.title('T20')
    #plt.savefig('T20_PxM.pdf')
    plt.show()
#        plot.legend()
    


def T30(h,            
        Fs = default['samplingRate'],       
        init = -5,
        end = -35,
        factor = 2,
        t30 = np.zeros(33)):
        
            raw_signal = h.timeSignal[:]
            raw_signal = np.array(raw_signal)
            octfilter = FractionalOctaveFilterbank(sample_rate=44100, order=4, nth_oct=3.0, norm_freq=1000, start_band=-19, end_band=13, edge_correction_percent=0.01, filterfun='py')
        
            hfilt = octfilter.filter(raw_signal)
            filtered_signal = hfilt[0]
            f_3rd = np.fromiter(hfilt[1].keys(), dtype = float)
            t30_result = [t30, f_3rd]
            
            for band in range(33):
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
                x = np.arange(init_sample, end_sample + 1) / Fs
                y = sch_db[init_sample: end_sample + 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        
            # Reverberation time (T30, T20, T10 or EDT)
                db_regress_init = (init - intercept) / slope
                db_regress_end = (end - intercept) / slope
                t30[band] = factor*((end_sample/44100)-(init_sample/44100))
                t30_result = [t30, f_3rd]
            return t30_result


def plot_T30(t30_result):
    """
    Frequency domain plotting method
    """
    plt.figure(figsize=(10, 5))
    p1 = plt.semilogx(t30_result[1], t30_result[0], 'k--')
    #p2 = plt.semilogx(f2,t20_m, 'b--')
    #plt.legend((p1[0], p2[0]),('Python','Matlab'))
    plt.xscale('log')
    plt.axis([100, 5000, 0, 1.1*np.max(t30_result[0])])
    plt.xlabel(r'$F$ in Hz')
    plt.ylabel(r'$T30$')
    plt.xticks(t30_result[1], ('100', '125', '160', '200', '250', '314', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000', '10000'))
    plt.title('T30')
    #plt.savefig('T20_PxM.pdf')
    plt.show()
#        plot.legend()

def EDT(h,            
        Fs = default['samplingRate'],       
        init = 0,
        end = -10,
        factor = 6,
        EDT = np.zeros(33)):

            raw_signal = h.timeSignal[:]
            raw_signal = np.array(raw_signal)        
    
            octfilter = FractionalOctaveFilterbank(sample_rate=44100, order=4, nth_oct=3.0, norm_freq=1000, start_band=-19, end_band=13, edge_correction_percent=0.01, filterfun='py')
            
            hfilt = octfilter.filter(raw_signal)
            filtered_signal = hfilt[0]
            
            f_3rd = np.fromiter(hfilt[1].keys(), dtype = float)
            
            EDT_result = [EDT, f_3rd]
            
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
                x = np.arange(init_sample, end_sample + 1) / Fs
                y = sch_db[init_sample: end_sample + 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            
                    # Reverberation time (T30, T20, T10 or EDT)
                db_regress_init = (init - intercept) / slope
                db_regress_end = (end - intercept) / slope
                EDT[band] = factor*((end_sample/44100))
                EDT_result = [EDT, f_3rd]
        
            return EDT_result
        
def plot_EDT(EDT_result):
    """
    Frequency domain plotting method
    """
    plt.figure(figsize=(10, 5))
    p1 = plt.semilogx(EDT_result[1], EDT_result[0], 'k--')
    #p2 = plt.semilogx(f2,t20_m, 'b--')
    #plt.legend((p1[0], p2[0]),('Python','Matlab'))
    plt.xscale('log')
    plt.axis([100, 5000, 0, 1.1*np.max(t20_result[0])])
    plt.xlabel(r'$F$ in Hz')
    plt.ylabel(r'$EDT$')
    plt.xticks(t20_result[1], ('100', '125', '160', '200', '250', '314', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000', '10000'))
    plt.title('EDT')
    #plt.savefig('T20_PxM.pdf')
    plt.show()
#        plot.legend()