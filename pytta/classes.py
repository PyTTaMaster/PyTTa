#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes
========
  
@Autores:
- Matheus Lazarin Alberto, mtslazarin@gmail.com
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

This submodule is mainly the means to an end. PyTTa is made intended to be
user friendly, the manipulation of the classes are documented here, but their
instantiation should be used through the <generate> submodule:
    
    >>> pytta.generate.sweep()
    >>> pytta.generate.noise()
    >>> pytta.generate.measurement('playrec')
    >>> pytta.generate.measurement('rec', lengthDomain = 'time', timeLen = 5)
    
This way, the default settings will be loaded into any object instantiated.

User intended classes:
    
    >>> pytta.SignalObj()
    >>> pytta.RecMeasure()
    >>> pytta.PlayRecMeasure()
    >>> pytta.FRFMeasure()
    
For further information see the specific class, or method, documentation
"""
#%% Importing modules
#import pytta as pa
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.lines as mlines
import scipy.signal as signal
import sounddevice as sd
from pytta import default
import time


class PyTTaObj(object):
    """
    PyTTa object class to define some properties and methods to be used 
    by any signal and processing classes. pyttaObj is a private class created
    just to shorten attributes declaration to each PyTTa class.
    
    Properties(self): (default), (dtype), meaning;
    
        - samplingRate: (44100), (int), signal's sampling rate;
        - lengthDomain: ('time'), (str), input array's domain. May be 'time' or 'samples';
        - timeLength: (seconds), (float), signal's time length in seconds;
        - fftDegree: (fftDegree), (float), 2**fftDegree signal's number of samples;
        - numSamples: (samples), (int), signal's number of samples
        - freqMin: (20), (int), minimum frequency bandwidth limit;
        - freqMax: (20000), (int), maximum frequency bandwidth limit;
        - comment: ('No comments.'), (str), some commentary about the signal or measurement object;
        
    """

    def __init__(self,
                 samplingRate=None,
                 lengthDomain=None,
                 fftDegree = None,
                 timeLength = None,
                 numSamples = None,
                 freqMin = None,
                 freqMax = None,
                 comment = "No comments."
                 ):

        self._lengthDomain = lengthDomain
        self._samplingRate = samplingRate
        self._fftDegree = fftDegree
        self._timeLength = timeLength
        self._numSamples = numSamples
        if freqMin == None or freqMax == None:
            self._freqMin, self._freqMax = default.freqMin, default.freqMax
        else:
            self._freqMin, self._freqMax = freqMin, freqMax
        self._comment = comment

#%% PyTTaObj Properties

    @property
    def samplingRate(self):
        return self._samplingRate

    @samplingRate.setter
    def samplingRate(self,newSamplingRate):
        self._samplingRate = newSamplingRate

    @property
    def lengthDomain(self):
        return self._lengthDomain
                
    @lengthDomain.setter
    def lengthDomain(self,newDomain):
        self._lengthDomain = newDomain

    @property
    def fftDegree(self):
        return self._fftDegree

    @fftDegree.setter
    def fftDegree(self,newFftDegree):
        self._fftDegree = newFftDegree

    @property
    def timeLength(self):
        return self._timeLength

    @timeLength.setter
    def timeLength(self,newTimeLength):
        self._timeLength = newTimeLength

    @property
    def numSamples(self):
        return self._numSamples

    @numSamples.setter
    def numSamples(self,newNumSamples):
        self._numSamples = newNumSamples

    @property
    def freqMin(self):
        return self._freqMin

    @freqMin.setter
    def freqMin(self,newFreqMin):
        self._freqMin = newFreqMin

    @property
    def freqMax(self):
        return self._freqMax

    @freqMax.setter
    def freqMax(self,newFreqMax):
        self._freqMax = newFreqMax

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self,newComment):
        self._comment = newComment

#%% PyTTaObj Methods

    def __call__(self):
        for name, value in vars(self).items():
            if len(name)<=8:
                print(name[1:]+'\t\t =',value)
            else: 
                print(name[1:]+'\t =',value)
                
    

class SignalObj(PyTTaObj):

    """
    Signal object class.
    
    Properties(self): (default), (dtype), meaning;
    
        - domain: ('time'), (str) domain of the input array;
        - timeSignal: (ndarray), (NumPy array), signal at time domain;
        - timeVector: (ndarray), (NumPy array), time reference vector for timeSignal;
        - freqSignal: (ndarray), (NumPy array), signal at frequency domain;
        - freqVector: (ndarray), (NumPy array), frequency reference vector for freqSignal;
        - unit: (None), (str), signal's unit. May be 'V' or 'Pa';
        - channelName: (dict), (dict/str), channels name dict;
        
    Properties(inherited): (default), (dtype), meaning;
    
        - samplingRate: (44100), (int), signal's sampling rate;
        - lengthDomain: ('time'), (str), input array's domain. May be 'time' or 'samples';
        - timeLength: (seconds), (float), signal's duration;
        - fftDegree: (fftDegree), (float), 2**fftDegree signal's number of samples;
        - numSamples: (samples), (int), signal's number of samples
        - freqMin: (20), (int), minimum frequency bandwidth limit;
        - freqMax: (20000), (int), maximum frequency bandwidth limit;
        - comment: ('No comments.'), (str), some commentary about the signal or measurement object;
        
    Methods: meaning;
    
        - play(): reproduce the timeSignal with default output device;
        - plot_time(): generates the signal's historic graphic;
        - plot_freq(): generates the signal's spectre graphic;
        - calibVoltage(): voltage calibration from an input SignalObj;
        - calibPressure(): pressure calibration from an input SignalObj;
    
    """
    
    def __init__(self,
                     signalArray=np.array([0],ndmin=2).T,
                     domain='time',
                     unit=None,
                     channelName=None,
                     *args,
                     **kwargs):
        if self.size_check(signalArray)>2:
            message = "No 'pyttaObj' is able handle arrays with more \
                        than 2 dimensions, '[:,:]', YET!."
            raise AttributeError(message)
        else:
            pass
        if self.size_check(signalArray) == 1:
            signalArray = np.array(signalArray,ndmin=2).T
        super().__init__(*args,**kwargs)
        # domain and initializate stuff
        self.domain = domain or args[1]
        if self.domain == 'freq':
            self.freqSignal = signalArray # [-] signal in frequency domain
        elif self.domain == 'time':
            self.timeSignal = signalArray # [-] signal in time domain
        else:
            self.timeSignal = signalArray
            print('Taking the input as a time domain signal')
            self.domain = 'time'
        self.unit = unit
        self.channelName = channelName
        self.CF = {}
            

#%% SignalObj Properties
    
    @property 
    def timeVector(self):
        return self._timeVector
    
    @property 
    def freqVector(self):
        return self._freqVector
            
    @property # when timeSignal is called returns the ndarray
    def timeSignal(self):
        return self._timeSignal
    
    @timeSignal.setter
    def timeSignal(self,newSignal): # when timeSignal have new ndarray value,
                                    # calculate other properties
        if self.size_check(newSignal) == 1:
            newSignal = np.array(newSignal,ndmin=2).T
        self._timeSignal = np.array(newSignal)
        self._numSamples = len(self.timeSignal) # [-] number of samples
        self._fftDegree = np.log2(self.numSamples) # [-] size parameter
        self._timeLength = self.numSamples / self.samplingRate # [s] signal time lenght
        self._timeVector = np.linspace(0,self.timeLength - (1/self.samplingRate),
                                      self.numSamples ) # [s] time vector (x axis)
        self._freqVector = np.linspace(0,(self.numSamples - 1) * self.samplingRate / (2*self.numSamples),
                                      (self.numSamples/2)+1 if self.numSamples%2==0 else (self.numSamples+1)/2 ) # [Hz] frequency vector (x axis)
        self._freqSignal = np.fft.rfft(self.timeSignal,axis=0,norm=None) # [-] signal in frequency domain

    @property
    def freqSignal(self): 
        return self._freqSignal
    
    @freqSignal.setter
    def freqSignal(self,newSignal):
        if self.size_check(newSignal) == 1:
            newSignal = np.array(newSignal,ndmin=2).T
        self._freqSignal = np.array(newSignal)
        self._timeSignal =  np.fft.irfft(self.freqSignal,axis=0,norm=None)
        self._numSamples = len(self.timeSignal) # [-] number of samples
        self._fftDegree = np.log2(self.numSamples) # [-] size parameter
        self._timeLength = self.numSamples/self.samplingRate  # [s] signal time lenght
        self._timeVector = np.arange(0,self.timeLength, 1/self.samplingRate) # [s] time vector
        self._freqVector = np.linspace(0,(self.numSamples-1) * self.samplingRate / (2*self.numSamples),
                                       (self.numSamples/2)+1 if self.numSamples%2==0  else (self.numSamples+1)/2 ) # [Hz] frequency vector

    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self,newunit):
        if newunit == 'V':
            self._unit = newunit
            self.dBName = 'dBu'
            self.dBRef = 0.775
        elif newunit == 'Pa':
            self._unit = newunit
            self.dBName = 'dB(z)'
            self.dBRef = 2e-5
        elif newunit == 'W/m2':
            self._unit = newunit
            self.dBName = 'dB'
            self.dBRef = 1e-12
        elif newunit == None:
            self._unit = ''
            self.dBName = 'dBFs'
            self.dBRef = 1
        else:
            raise TypeError(newunit+' unit not accepted. May be Pa, V or None.')

    @property
    def channelName(self):
        return self._channelName
    
    @channelName.setter
    def channelName(self,channelName):
        if channelName == None:
            self._channelName = []           
            for chIndex in range(0,self.num_channels()):
                self._channelName.append('Channel '+str(chIndex+1))
        elif len(channelName) == self.num_channels():
            self._channelName = []   
            self._channelName = channelName
        else:
            raise AttributeError('Incompatible number of channel names and channel number.')
            
#%% SignalObj Methods
        
    def __truediv__(self, other):
        """
        Frequency domain division method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike")
        result = SignalObj(samplingRate=self.samplingRate)
        result._domain = 'freq'
        if self.size_check() > 1:
            if other.size_check() > 1:
                i = 0
                for channelA in range(self.num_channels()):
                    for channelB in range(other.num_channels()):
                        result.freqSignal[:,i + channelB] = \
                                self.freqSignal[:,channelA] / other.freqSignal[:,channelB]
                    i = channelB
            else:
                for channel in range(self.num_channels()):
                    result.freqSignal = self.freqSignal[:,channel] / other.freqSignal
        elif other.size_check() > 1:
            for channel in range(self.num_channels()):
                result.freqSignal = self.freqSignal / other.freqSignal[:,channel]
        else: result.freqSignal = self.freqSignal / other.freqSignal        
        return result
    
    
    def __add__(self, other):
        """
        Time domain addition method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike")
        result = SignalObj(samplingRate=self.samplingRate)
        result.domain = 'time'
        if self.size_check() > 1:
            if other.size_check() > 1:
                i = 0
                for channelA in range(self.num_channels()):
                    for channelB in range(other.timeSignal.shape):
                        result.timeSignal[:,i + channelB] = \
                                self.timeSignal[:,channelA] + other.timeSignal[:,channelB]
                    i = channelB
            else:
                for channel in range(self.num_channels()):
                    result.freqSignal = self.timeSignal[:,channel] + other.timeSignal
        elif other.size_check() > 1:
            for channel in range(self.num_channels()):
                result.timeSignal = self.timeSignal + other.timeSignal[:,channel]
        else: result.timeSignal = self.timeSignal + other.timeSignal
        return result

    def __sub__(self, other):
        """
        Time domain subtraction method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike")

        result = SignalObj(samplingRate=self.samplingRate)
        result.domain = 'time'
        if self.size_check() > 1:
            if other.size_check() > 1:
                i = 0
                for channelA in range(self.num_channels()):
                    for channelB in range(other.num_channels):
                        result.timeSignal[:,i + channelB] = \
                        self.timeSignal[:,channelA] - other.timeSignal[:,channelB]
                    i = channelB
            else:
                for channel in range(self.num_channels()):
                    result.freqSignal = self.timeSignal[:,channel] - other.timeSignal
                                        
        elif other.size_check() > 1:
            for channel in range(self.num_channels()):
                result.timeSignal = self.timeSignal - other.timeSignal[:,channel]
        else: result.timeSignal = self.timeSignal - other.timeSignal
        return result

    def mean(self):
        return SignalObj(signalArray=np.mean(self.timeSignal,1),lengthDomain='time',samplingRate=self.samplingRate)
    
    def num_channels(self):
        try:
            numChannels = np.shape(self.timeSignal)[1]
        except IndexError:
            numChannels = 1
        return numChannels
    
    def max_level(self):
        maxlvl = []
        for chIndex in range(self.num_channels()):
            maxlvl.append(np.max(np.abs(self.timeSignal[:,chIndex])))
        maxlvl = np.array(maxlvl)
        maxlvl = 20*np.log10(maxlvl/self.dBRef)
        return maxlvl
            
    
    def size_check(self, inputArray = []):
        if inputArray == []: inputArray = self.timeSignal[:]
        return np.size( inputArray.shape )


    def play(self,outChannel=None,latency='low',**kwargs):
        """
        Play method
        """
        if outChannel == None:
            if self.num_channels() <=1:
                outChannel = default.outChannel
            elif self.num_channels() > 1:
                outChannel = np.arange(1,self.num_channels()+1)
                
        sd.play(self.timeSignal,self.samplingRate,mapping=outChannel,**kwargs)
			   
#   def plot(self): # TODO
#        ...

    def plot_time(self):
        """
        Time domain plotting method
        """
        # DB
        plot.figure( figsize=(10,5) )
        if self.num_channels() > 1:
            for chIndex in range(self.num_channels()):
                plot.plot( self.timeVector, self.timeSignal[:,chIndex],label=self.channelName[chIndex])            
        else:
            plot.plot( self.timeVector, self.timeSignal,label=self.channelName[0])            
        plot.legend(loc='best')
        plot.grid(color='gray', linestyle='-.', linewidth=0.4)
        plot.axis( ( self.timeVector[0] - 10/self.samplingRate, \
                    self.timeVector[-1] + 10/self.samplingRate, \
                    1.05*np.min( self.timeSignal ), \
                   1.05*np.max( self.timeSignal ) ) )
        plot.xlabel(r'$Time$ [s]')
        plot.ylabel(r'$Amplitude$ ['+self.unit+']')
        
    def plot_freq(self,smooth=False):
        """
        Frequency domain plotting method
        """
        plot.figure( figsize=(10,5) )
        if not smooth:
            if self.num_channels() > 1:
                for chIndex in range(0,self.num_channels()):
                    dBSignal = 20 * np.log10( (1 / len(self.freqSignal) ) * np.abs( self.freqSignal[:,chIndex]) / self.dBRef )
                    plot.semilogx( self.freqVector,dBSignal,label=self.channelName[chIndex])
            else:
                dBSignal = 20 * np.log10( (1 / len(self.freqSignal) ) * np.abs( self.freqSignal) / self.dBRef )
                plot.semilogx( self.freqVector, dBSignal ,label=self.channelName[0])            
        else:
            if self.num_channels() > 1:
                for chIndex in range(self.num_channels()):
                    signalSmooth = signal.savgol_filter( np.abs(self.freqSignal[:,chIndex]), 31, 3 )
                    dBSignal = 20 * np.log10( (1 / len(self.freqSignal) ) * np.abs( signalSmooth ) / self.dBRef )
                    plot.semilogx( self.freqVector, dBSignal ,label=self.channelName[chIndex])
            else:
                signalSmooth = signal.savgol_filter( np.squeeze(np.abs(self.freqSignal)), 31, 3 )
                dBSignal = 20 * np.log10( (2 / self.numSamples ) * np.abs( signalSmooth ) / self.dBRef )
                plot.semilogx( self.freqVector, dBSignal ,label=self.channelName[0])
        plot.grid(color='gray', linestyle='-.', linewidth=0.4)        
        plot.legend(loc='best')
        if np.max(dBSignal) > 0:
            ylim = [np.max(dBSignal)-80,1.12*np.max(dBSignal)]
        else:
            ylim = [1/1.05*np.min(dBSignal),1/1.05*np.max(dBSignal)]
        plot.axis((self.freqMin,self.freqMax,ylim[0],ylim[1]))
        plot.xlabel(r'$Frequency$ [Hz]')
        plot.ylabel(r'$Magnitude$ ['+self.dBName+' ref.: '+str(self.dBRef)+'['+self.unit+']')

    def calibVoltage(self,refSignalObj,referenceVoltage):
        """
        calibVoltage method: use informed SignalObj with a calibration voltage signal, and the reference RMS voltage to calculate the Correction Factor.
        
            >>> SignalObj.calibVoltage(refSignalObj,referenceVoltage)
            
        """
        self.referenceVoltage = referenceVoltage
        self.refSignal = refSignalObj
        rms = (np.mean(refSignalObj.timeSignal[:,0]**2))**(1/2)
        self.CF['V'] = self.referenceVoltage/rms
        if self.unit != 'Pa':
            self.unit = 'V'
        self.timeSignal = self.timeSignal*self.CF['V']
        
    def calibPressure(self,refSignalObj,referencePressure):
        """
        calibPressure method: use informed SignalObj, with a calibration acoustic pressure signal, and the reference RMS acoustic pressure to calculate the Correction Factor.
        
            >>> SignalObj.calibPressure(refSignalObj,referencePressure)
            
        """
        self.referencePressure = referencePressure
        self.refSignal = refSignalObj
        rms = (np.mean(refSignalObj.timeSignal[:,0]**2))**(1/2)
        self.CF['Pa'] = self.referencePressure/rms
        self.unit = 'Pa'
        self.timeSignal = self.timeSignal*self.CF['Pa']


#%% Measurement class
class Measurement(PyTTaObj):
    """
    Measurement object class created to define some properties and methods to
    be used by the playback, recording and processing classes. It is a private
    class
    
    Properties(self): (default), (dtype), meaning;
    
        - device: (system default), (list/int), list of input and output devices;
        - inChannel: ([1]), (list/int), list of device's input channel used for recording;
        - outChannel: ([1]), (list/int), list of device's output channel used for playing/reproducing a signalObj;
        - calibratedChain: (0), (int), 1 if you want a calibrated measurement chain or 0 if you don't want;
        - vCalibratedCh: ([]), (list/int), list of voltage calibrated channels;
        - vCalibrationCF: ([0]), (list/float), list of correction factors for the voltage calibrated channels;

    Properties(inherited): 	(default), (dtype), meaning;

        - samplingRate: (44100), (int), signal's sampling rate;
        - lengthDomain: ('time'), (str), input array's domain. May be 'time' or 'samples';
        - timeLength: (seconds), (float), signal's time length in seconds;
        - fftDegree: (fftDegree), (float), 2**fftDegree signal's number of samples;
        - numSamples: (samples), (int), signal's number of samples
        - freqMin: (20), (int), minimum frequency bandwidth limit;
        - freqMax: (20000), (int), maximum frequency bandwidth limit;
        - comment: ('No comments.'), (str), some commentary about the signal or measurement object;
        
    """
    def __init__(self,
                 device=None,
                 inChannel=None,
                 outChannel=None,
                 calibratedChain=None,
                 channelName=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args,**kwargs)
        self.device = device # device number. For device list use sounddevice.query_devices()
        self.inChannel = inChannel # input channels
        self.outChannel = outChannel # output channels
        self.calibratedChain = 0 if calibratedChain == None else calibratedChain # optin for a calibrated chain
        self.channelName = channelName
        self.vCalibratedCh = [] # list of calibrated channels
        self.vCalibrationCF = [] # list of calibration correction factors
        
#%% Measurement Properties
        
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self,newDevice):
        self._device = newDevice
    
    @property
    def inChannel(self):
        return self._inChannel
    
    @inChannel.setter
    def inChannel(self,newInputChannel):
        if not isinstance(newInputChannel,list):
            raise AttributeError('inChannel must be a list; e.g. [1] .')
        else:
            try:    
                oldChName = self.channelName            
                oldInCh = self.inChannel
                self._inChannel = newInputChannel
                self.channelName = None
                for i in oldInCh:
                    if i in self._inChannel:
                        self.channelName[self._inChannel.index(i)] = oldChName[oldInCh.index(i)]
            except AttributeError:
                self._inChannel = newInputChannel
            
    @property
    def outChannel(self):
        return self._outChannel

    @outChannel.setter
    def outChannel(self,newOutputChannel):
#        if not isinstance(newOutputChannel,list): # BUG WHEN CREATE A REC MEASUREMENT WITHOUT OUTCHANNEL
#            raise AttributeError('outChannel must be a list; e.g. [1] .')
#        else:
        self._outChannel = newOutputChannel
        
    @property
    def calibratedChain(self):
        return self._calibratedChain
    
    @calibratedChain.setter
    def calibratedChain(self,newoption):
        if newoption not in range(2):
            raise AttributeError('calibratedChain must be 1 for a calibrated measurement chain or 0 for a non calibrated.')
        else:
            self._calibratedChain = newoption
            
    @property
    def channelName(self):
        return self._channelName
    
    @channelName.setter
    def channelName(self,channelName):
        if channelName == None:
            self._channelName = []           
            for chIndex in range(0,len(self.inChannel)):
                self._channelName.append('Channel '+str(chIndex+1))
        elif len(channelName) == len(self.inChannel):
            self._channelName = []   
            self._channelName = channelName
        else:
            raise AttributeError('Incompatible number of channel names and channel number.')
            
#%% Measurement Methods
        
    def calibVoltage(self,referenceVoltage=None,channel=None):
        """
        calibVoltage method: acquire the informed calibration voltage signal and calculates the Correction Factor to the specified channel.
        
            >>> pytta.PlayRecMeasure.calibVoltage(referenceVoltage,channel)
        """
        if self.calibratedChain != 1: self.calibratedChain = 1
        if channel not in self.inChannel:
            raise AttributeError(str(channel) + ' is not a valid input channel for this measurement. Maybe you need to add it in inChannel property.')
        else:
            pass
        self.referenceVoltage = referenceVoltage
        vCalibrationRec = sd.rec(3*self.samplingRate,
                                            self.samplingRate,
                                            mapping = channel,
                                            blocking=True,
                                            latency='low',
                                            dtype = 'float32')
        rms = (np.mean(vCalibrationRec**2))**(1/2)
        CF = self.referenceVoltage/rms
        self.vCalibrationCF.append(CF)
        self.vCalibratedCh.append(channel)

#%% RecMeasure class        
class RecMeasure(Measurement):
    """
    Signal Recording object
    
    Properties(self): (default), (dtype), meaning:
        
        - lengthDomain: ('time'), (str), input array's domain. May be 'time' or 'samples';
        - timeLength: (seconds), (float), signal's time length in seconds;
        - fftDegree: (fftDegree), (float), 2**fftDegree signal's number of samples;

    Properties(inherited): (default), (dtype), meaning;
    
        - device: (system default), (list/int), list of input and output devices;
        - inChannel: ([1]), (list/int), list of device's input channel used for recording;
        - calibratedChain: (0), (int), 1 if you want a calibrated measurement chain or 0 if you don't want;
        - vCalibratedCh: ([]), (list/int), list of voltage calibrated channels;
        - vCalibrationCF: ([0]), (list/float), list of correction factors for the voltage calibrated channels;
        - samplingRate: (44100), (int), signal's sampling rate;
        - numSamples: (samples), (int), signal's number of samples
        - freqMin: (20), (int), minimum frequency bandwidth limit;
        - freqMax: (20000), (int), maximum frequency bandwidth limit;
        - comment: ('No comments.'), (str), some commentary about the signal or measurement object;      

	Methods: meaning;
    
		- run(): starts recording using the inch and device information, during timeLen seconds;
        
    """
    def __init__(self,lengthDomain=None,
                 fftDegree=None,
                 timeLength=None,
                 *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lengthDomain = lengthDomain
        if self.lengthDomain == 'samples':
            self._fftDegree = fftDegree
        elif self.lengthDomain == 'time':
            self._timeLength = timeLength
        else:
            self._timeLength = None
            self._fftDegree = None

#%% Rec Properties
            
    @property
    def timeLength(self):
        return self._timeLength
    
    @timeLength.setter
    def timeLength(self,newLength):
        self._timeLength = np.round( newLength, 2 )
        self._numSamples = self.timeLength * self.samplingRate
        self._fftDegree = np.round( np.log2( self.numSamples ), 2 )
        
    @property
    def fftDegree(self):
        return self._fftDegree
    
    @fftDegree.setter
    def fftDegree(self,newDegree):
        self._fftDegree = np.round( newDegree, 2 )
        self._numSamples = 2**self.fftDegree
        self._timeLength = np.round( self.numSamples / self.samplingRate, 2 )

#%% Rec Methods
        
    def run(self):
        """
        Run method: starts recording during Tmax seconds
        Outputs a signalObj with the recording content
        """
        # Checking if all channels are cailabrated if calibratedChain is setted to 1
        if self.calibratedChain == 1:
            calibratedCheck = []
            notCalibratedCh = []            
            for chindex in range(0,len(self.inChannel)):
                if self.inChannel[chindex] not in self.vCalibratedCh:
                    calibratedCheck.append(0)
                    notCalibratedCh.append(self.inChannel[chindex])
                else:
                    calibratedCheck.append(1)                    
            if 0 in calibratedCheck:
                raise AttributeError('calibratedChain is set to 1. You must calibrated the following remain channels: ' + str(notCalibratedCh))                
        # Record
        self.recording = sd.rec(self.numSamples,
                                self.samplingRate,
                                mapping = self.inChannel,
                                blocking=True,
                                device=self.device,
                                latency='low',
                                dtype = 'float32')
        self.recording = np.squeeze(self.recording)
        self.recording = SignalObj(signalArray=self.recording,domain='time',samplingRate=self.samplingRate)        
        # Apply the calibration Correction Factor
        if self.calibratedChain == 1:
            if self.recording.size_check() > 1:
                for chindex in range(self.recording.num_channels()):
                    self.recording.timeSignal[:,chindex] = self.vCalibrationCF[chindex]*self.recording.timeSignal[:,chindex]
                self.recording.unit = 'V'
            else:
                self.recording.timeSignal[:] = self.vCalibrationCF*self.recording.timeSignal[:]
                self.recording.unit = 'V'
        self.recording.channelName = self.channelName
        self.recording.timeStamp = time.ctime(time.time())
        maxOut = np.max(np.abs(self.recording.timeSignal))
        print('max input level (recording): ',20*np.log10(maxOut/self.recording.dBRef),' ',self.recording.dBName,' - ref.: ',str(self.recording.dBRef),' [',self.recording.unit,']')
        self.recording.freqMin, self.recording.freqMax = (self.freqMin,self.freqMax)
        return self.recording    
    
#%% PlayRecMeasure class 
class PlayRecMeasure(Measurement):
    """
    Playback and Record object

    Properties(self), (default), (dtype), meaning:
        
		- excitation: (SignalObj), (SignalObj), signal information used to reproduce (playback);

    Properties(inherited): 	(default), (dtype), meaning;
    
        - device: (system default), (list/int), list of input and output devices;
        - inChannel: ([1]), (list/int), list of device's input channel used for recording;
        - outChannel: ([1]), (list/int), list of device's output channel used for playing/reproducing a signalObj;
        - calibratedChain: (0), (int), 1 if you want a calibrated measurement chain or 0 if you don't want;
        - vCalibratedCh: ([]), (list/int), list of voltage calibrated channels;
        - vCalibrationCF: ([0]), (list/float), list of correction factors for the voltage calibrated channels;
        - samplingRate: (44100), (int), signal's sampling rate;
        - lengthDomain: ('time'), (str), input array's domain. May be 'time' or 'samples';
        - timeLength: (seconds), (float), signal's time length in seconds;
        - fftDegree: (fftDegree), (float), 2**fftDegree signal's number of samples;
        - numSamples: (samples), (int), signal's number of samples
        - freqMin: (20), (int), minimum frequency bandwidth limit;
        - freqMax: (20000), (int), maximum frequency bandwidth limit;
        - comment: ('No comments.'), (str), some commentary about the signal or measurement object;
        
	Methods: meaning;
		- run(): starts playing the excitation signal and recording during the excitation timeLen duration;

    """
    def __init__(self,excitation=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if excitation is None:
            self._excitation = None
        else:
            self.excitation = excitation
#%% PlayRec Methods
            
    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Outputs a signalObj with the recording content
        """
        timeStamp = time.ctime(time.time())
        recording = sd.playrec(self.excitation.timeSignal,
                             samplerate=self.samplingRate, 
                             input_mapping=self.inChannel,
                             output_mapping=self.outChannel,
                             device=self.device,
                             blocking=True,
                             latency='low',
                             dtype = 'float32'
                             ) # y_all(t) - out signal: x(t) conv h(t)
        recording = np.squeeze( recording ) # turn column array into line array
        self.recording = SignalObj(signalArray=recording,domain='time',samplingRate=self.samplingRate )
        self.recording.timeStamp = timeStamp
        for chIndex in range(self.excitation.num_channels()):
            print('max output level (excitation) on channel ['+str(chIndex+1)+']: '+str(self.excitation.max_level()[chIndex])+' '+self.excitation.dBName+' - ref.: '+str(self.excitation.dBRef)+' ['+self.excitation.unit+']')
        for chIndex in range(self.recording.num_channels()):
            print('max input level (recording): on channel ['+str(chIndex+1)+']: '+str(self.recording.max_level()[chIndex])+' '+self.recording.dBName+' - ref.: '+str(self.recording.dBRef)+' ['+self.recording.unit+']')
        return self.recording

#%% PlayRec Properties
            
    @property
    def excitation(self):
        return self._excitation        
    @excitation.setter
    def excitation(self,newSignalObj):
        self._excitation = newSignalObj
        
    @property
    def samplingRate(self):
        return self.excitation._samplingRate

    @property
    def fftDegree(self):
        return self.excitation._fftDegree

    @property
    def timeLength(self):
        return self.excitation._timeLength

    @property
    def numSamples(self):
        return self.excitation._numSamples

    @property
    def freqMin(self):
        return self.excitation._freqMin

    @property
    def freqMax(self):
        return self.excitation._freqMax


#%% FRFMeasure class     
class FRFMeasure(PlayRecMeasure):
    """
    Transferfunction object
        
    Properties(inherited): 	(default), (dtype), meaning;
    
        - excitation: (SignalObj), (SignalObj), signal information used to reproduce (playback);
        - device: (system default), (list/int), list of input and output devices;
        - inChannel: ([1]), (list/int), list of device's input channel used for recording;
        - outChannel: ([1]), (list/int), list of device's output channel used for playing/reproducing a signalObj;
        - calibratedChain: (0), (int), 1 if you want a calibrated measurement chain or 0 if you don't want;
        - vCalibratedCh: ([]), (list/int), list of voltage calibrated channels;
        - vCalibrationCF: ([0]), (list/float), list of correction factors for the voltage calibrated channels;
        - samplingRate: (44100), (int), signal's sampling rate;
        - lengthDomain: ('time'), (str), input array's domain. May be 'time' or 'samples';
        - timeLength: (seconds), (float), signal's time length in seconds;
        - fftDegree: (fftDegree), (float), 2**fftDegree signal's number of samples;
        - numSamples: (samples), (int), signal's number of samples
        - freqMin: (20), (int), minimum frequency bandwidth limit;
        - freqMax: (20000), (int), maximum frequency bandwidth limit;
        - comment: ('No comments.'), (str), some commentary about the signal or measurement object;
        
	Methods: meaning;
    
		- run(): starts playing the excitation signal and recording during the excitation timeLen duration;

    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Divides the recorded signalObj by the excitation signalObj to generate a transferfunction
        Outputs the transferfunction signalObj
        """
        self.recording = super().run()
        self.transferfunction = self.recording/self.excitation
        self.transferfunction.timeStamp = self.recording.timeStamp
        return self.transferfunction
    