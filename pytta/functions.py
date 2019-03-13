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
from .classes import SignalObj

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
    Reads a wave file into a SignalObj   
    """
    samplingRate, data = wf.read(fileName)
    signal = SignalObj(data,'time',samplingRate)
    return signal

def write_wav(fileName,signalIn):
    """
    Writes a SignalObj into a single wave file
    """
    samplingRate = signalIn.samplingRate
    data = signalIn.timeSignal
    return wf.write(fileName,samplingRate,data)


def merge(signal1,*signalObjects):
    """
    Gather all of the input argument signalObjs into a single
    signalObj and place the respective timeSignal of each
    as a column of the new object
    """
    mergedSignal = signal1.timeSignal
    numSamples = signal1.numSamples
    samplingRate = signal1.samplingRate
    k = 1
    for inObj in signalObjects:
        mergedSignal = np.append(mergedSignal[:],inObj.timeSignal[:])
        k += 1
    mergedSignal = np.array(mergedSignal)
    mergedSignal.resize(k,numSamples)
    mergedSignal = mergedSignal.transpose()
    newSignal = SignalObj(mergedSignal,'time',samplingRate)
    return newSignal

def fft_convolve(signal1,signal2):
    """
    Uses scipy.signal.fftconvolve() to convolve two time domain signals.
    
    >>> convolution = pytta.fft_convolve(signal1,signal2)
    """
#    Fs = signal1.Fs
    conv = ss.fftconvolve(signal1.timeSignal,signal2.timeSignal)
    signal = SignalObj(conv, 'time', signal1.samplingRate)
    return signal

def find_delay(signal1, signal2):
    """
    Cross Correlation alternative, more efficient fft based method to calculate time shift between two signals.
   
    >>> shift = pytta.find_delay(signal1,signal2)
    """
    if signal1.N != signal2.N:
        return print('Signal1 and Signal2 must have the same length')
    else:
        freqSignal1 = signal1.freqSignal
        freqSignal2 = sfft.fft( np.flipud( signal2.timeSignal ) )
        convoluted = np.real( sfft.ifft( freqSignal1 * freqSignal2 ) )
        convShifted = sfft.fftshift( convoluted )
        zeroIndex = int(signal1.numSamples / 2) - 1
        shift = zeroIndex - np.argmax(convShifted)          
    return shift

def corr_coef(signal1, signal2):
    """
    Finds the correlation coeficient between two SignalObjs using
    the numpy.corrcoef() function.
    """
    coef = np.corrcoef(signal1.timeSignal, signal2.timeSignal)
    return coef[0,1]


def resample(signal,newSamplingRate):
    """
        Resample the timeSignal of the input SignalObj to the
        given sample rate using the scipy.signal.resample() function
    """
    newSignalSize = np.int(signal.timeLength*newSamplingRate)
    resampled = ss.resample(signal.timeSignal[:], newSignalSize)
    newSignal = SignalObj(resampled,"time",newSamplingRate)
    return newSignal