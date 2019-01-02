# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:05:00 2018

@author: Cecilia
"""
#import wave
from scipy.io import wavfile as wf
import numpy as np
import sounddevice as sd
import scipy.signal as ss
import scipy.fftpack as sfft
from .classes import signalObj

def list_devices():
    """
    list_devices()
    
        Shortcut to sounddevice.query_devices(). Made to exclude the need of
        importing Sounddevice directly just to find out which audio devices can
        be used.
		  
        >>> pytta.list_devices()
        
    """
    return sd.query_devices()


def read_wav(fileName):
	fs, data = wf.read(fileName)
	signal = signalObj(data,'time',fs)
	return signal

def write_wav(fileName,signalIn):
	Fs = signalIn.Fs
	data = signalIn.timeSignal
	return wf.write(fileName,Fs,data)


def merge(signal1,*signalObjects):
	mergedSignal = signal1.timeSignal
	N = signal1.N
	Fs = signal1.Fs
	k = 1
	for inObj in signalObjects:
		mergedSignal = np.append(mergedSignal[:],inObj.timeSignal[:])
		k += 1
	mergedSignal = np.array(mergedSignal)
	mergedSignal.resize(k,N)
	mergedSignal = mergedSignal.transpose()
	newSignal = signalObj(mergedSignal,'time',Fs)
	return newSignal

def fftconvolve(signal1,signal2):
    """
    fftconvolve()
        
        Uses scipy.signal.fftconvolve() to convolve two time domain signais.
        
        >>>convolution = pytta.fftconvolve(signal1,signal2)
    """
#    Fs = signal1.Fs
    conv = ss.fftconvolve(signal1.timeSignal,signal2.timeSignal)
    signal = signalObj(conv, 'time', signal1.Fs)
    return signal

def super_convolve(signal1,signal2):
    k1 = np.size(np.shape(signal1.timeSignal))
    k2 = np.size(np.shape(signal2.timeSignal))
	
    for arg in np.arange(0,k1):
        dummy1 = signalObj(signal1.timeSignal[:,arg], signal1.domain, signal1.Fs)
		
        if k2==1:
            if arg==0:
                conv = fftconvolve(dummy1,signal2)
            else:
                preConv = fftconvolve(dummy1,signal2)
                conv = merge(conv,preConv)
        else:
            for arg2 in np.arange(0,k2):
	            dummy2 = signalObj(signal2.timeSignal[:,arg2], signal2.domain, signal2.Fs)	
	            
	            if arg==0 and arg2==0:
	                conv = fftconvolve(dummy1,dummy2)
	            else:
	                preConv = fftconvolve(dummy1,dummy2)
	                conv = merge(conv,preConv)
    return conv

def finddelay(signal1, signal2):
    """
    finddelay()
        
       Cross Corrlation alternative, more efficient fft based method to calculate time shift between two signals.
       
       >>>shift = pytta.finddelay(signal1,signal2)
       
        
    """
    if signal1.N != signal2.N:
        return print('Signal1 and Signal2 must have the same length')
    else:
        f1 = signal1.freqSignal
        f2 = sfft.fft(np.flipud(signal2.timeSignal))
        cc = np.real(sfft.ifft(f1 * f2))
        ccs = sfft.fftshift(cc)
        zero_index = int(signal1.N / 2) - 1
        shift = zero_index - np.argmax(ccs)          
    return shift

def corrcoef(signal1, signal2):
    coef = np.corrcoef(signal1.timeSignal, signal2.timeSignal)
    return coef[0,1]


def resample(signal,newSamplingRate):
	newSignalSize = np.int(signal.timeLen*newSamplingRate)
	resampled = ss.resample(signal.timeSignal[:], newSignalSize)
	newSignal = signalObj(resampled,"time",newSamplingRate)
	return newSignal