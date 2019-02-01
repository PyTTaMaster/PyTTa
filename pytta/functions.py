# -*- coding: utf-8 -*-
"""
Functions
=========
    
    This submodule carries a set of useful functions of general purpouses when
    using PyTTa, like reading and writing wave files, seeing the audio IO devices
    available and some signal processing tools.
    
    Available functions:
    --------------------
    
        >>> pytta.list_devices()
        >>> pytta.read_wav( fileName )
        >>> pytta.write_wav( fileName, signalObject )
        >>> pytta.merge( signalObj1, signalObj2, ..., signalObjN )
        >>> pytta.fftconvolve( signalObj1, signalObj2 )
        >>> pytta.finddelay( signalObj1, signalObj2 )
        >>> pytta.corrcoef( signalObj1, signalObj2 )
        >>> pytta.resample( signalObj, newSamplingRate )
        
    For further information, check the function specific documentation.
"""

from scipy.io import wavfile as wf
import numpy as np
import sounddevice as sd
import scipy.signal as ss
import scipy.fftpack as sfft
from .classes import signalObj

def list_devices():
    """
    Shortcut to sounddevice.query_devices(). Made to exclude the need of
    importing Sounddevice directly just to find out which audio devices can
    be used.
		  
        >>> pytta.list_devices()
        
    """
    return sd.query_devices()


def read_wav(fileName):
    """
    Reads a wave file into a :class:signalObj   
    """
    fs, data = wf.read(fileName)
    signal = signalObj(data,'time',fs)
    return signal

def write_wav(fileName,signalIn):
    """
    Writes a :class:signalObj into a single wave file
    """
    Fs = signalIn.Fs
    data = signalIn.timeSignal
    return wf.write(fileName,Fs,data)


def merge(signal1,*signalObjects):
    """
    Gather all of the input argument signalObjs into a single
    signalObj and place the respective timeSignal of each
    as a column of the new object
    """
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
    Uses scipy.signal.fftconvolve() to convolve two time domain signais.
    
    >>>convolution = pytta.fftconvolve(signal1,signal2)
    """
#    Fs = signal1.Fs
    conv = ss.fftconvolve(signal1.timeSignal,signal2.timeSignal)
    signal = signalObj(conv, 'time', signal1.Fs)
    return signal

def finddelay(signal1, signal2):
    """
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
    """
    :func:corrcoef
    
        Finds the correlation coeficient between two :class:signalObjs using
        the numpy.corrcoef() function.
    """
    coef = np.corrcoef(signal1.timeSignal, signal2.timeSignal)
    return coef[0,1]


def resample(signal,newSamplingRate):
    """
    :func:resample
        
        Resample the :prop:timeSignal of the input :class:signalObj to the
        given sample rate using the scipy.signal.resample() function
    """
    newSignalSize = np.int(signal.timeLen*newSamplingRate)
    resampled = ss.resample(signal.timeSignal[:], newSignalSize)
    newSignal = signalObj(resampled,"time",newSamplingRate)
    return newSignal