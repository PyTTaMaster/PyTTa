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
        >>> pytta.fft_convolve( signalObj1, signalObj2 )
        >>> pytta.find_delay( signalObj1, signalObj2 )
        >>> pytta.corr_coef( signalObj1, signalObj2 )
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
    if data.dtype == 'int16': data = data/(2**15)
    if data.dtype == 'int32': data = data/(2**31)
    signal = SignalObj(data,'time',samplingRate=samplingRate)
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
    j=1;
    comment = signal1.comment
    channelName = signal1.channelName
    timeSignal = signal1.timeSignal
    for inObj in signalObjects:
        if signal1.samplingRate != inObj.samplingRate:
            message = '\
            \n To merge signals they must have the same sampling rate!\
            \n SignalObj 1 and '+str(j+1)+' have different sampling rates.'
            raise AttributeError(message)
        if signal1.numSamples != inObj.numSamples:
            message ='\
            \n To merge signals they must have the same length!\
            \n SignalObj 1 and '+str(j+1)+' have different lengths.'
            raise AttributeError(message)
        if signal1.unit != inObj.unit:
            message ='\
            \n To merge signals they must have the same unit!\
            \n SignalObj 1 and '+str(j+1)+' have different units.'
            raise AttributeError(message)            
        comment = comment + ' / ' + inObj.comment
        channelName = channelName + inObj.channelName
        timeSignal = np.hstack(( timeSignal, inObj.timeSignal ))
        j += 1
    newSignal = SignalObj(timeSignal,domain='time',samplingRate=signal1.samplingRate,channelName=channelName,comment=comment)
    return newSignal

def split(signal):
#    medDG = pytta.SignalObj(med.timeSignal[:,0],samplingRate=fs)
#medDL = pytta.SignalObj(med.timeSignal[:,1],samplingRate=fs)
    return 0

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