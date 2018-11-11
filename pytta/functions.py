# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:05:00 2018

@author: Cecilia
"""
import wave
from scipy.io import wavfile as wf
import numpy as np
import sounddevice as sd
import scipy.signal as ss
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

