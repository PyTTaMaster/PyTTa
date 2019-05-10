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

##%% Importing modules
#import pytta as pa
import numpy as np
import matplotlib.pyplot as plot
import scipy.signal as signal
import scipy.io as sio
import sounddevice as sd
from pytta import default
import time
import copy as cp


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
                 samplingRate = None,
                 lengthDomain = None,
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
        return

##%% PyTTaObj Properties
    @property
    def samplingRate(self):
        return self._samplingRate

    @samplingRate.setter
    def samplingRate(self,newSamplingRate):
        self._samplingRate = newSamplingRate
        return

    @property
    def lengthDomain(self):
        return self._lengthDomain
                
    @lengthDomain.setter
    def lengthDomain(self,newDomain):
        self._lengthDomain = newDomain
        return

    @property
    def fftDegree(self):
        return self._fftDegree

    @fftDegree.setter
    def fftDegree(self,newFftDegree):
        self._fftDegree = newFftDegree
        return

    @property
    def timeLength(self):
        return self._timeLength

    @timeLength.setter
    def timeLength(self,newTimeLength):
        self._timeLength = newTimeLength
        return

    @property
    def numSamples(self):
        return self._numSamples

    @numSamples.setter
    def numSamples(self,newNumSamples):
        self._numSamples = newNumSamples
        return

    @property
    def freqMin(self):
        return self._freqMin

    @freqMin.setter
    def freqMin(self,newFreqMin):
        self._freqMin = newFreqMin
        return

    @property
    def freqMax(self):
        return self._freqMax

    @freqMax.setter
    def freqMax(self,newFreqMax):
        self._freqMax = newFreqMax
        return

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self,newComment):
        self._comment = newComment
        return

##%% PyTTaObj Methods
    def __call__(self):
        for name, value in vars(self).items():
            if len(name)<=8:
                print(name[1:]+'\t\t =',value)
            else: 
                print(name[1:]+'\t =',value)
        return
    
    def save_mat(self,filename=time.ctime(time.time())):
        myObj = vars(self)
        for key, value in myObj.items():
            if value is None:
                myObj[key] = 0
            if isinstance(myObj[key],dict) and len(value) == 0:
                myObj[key] = 0
        myObjno_ = {}
        for key, value in myObj.items():
            if key.find('_') >= 0:
                key = key.replace('_','')
            myObjno_[key] = value
        sio.savemat(filename, myObjno_, format='5', oned_as='column')
        return


class ChannelObj(object):
    
    
    def __init__(self,
                 name='',
                 unit='',
                 CF=1,
                 calibCheck=False):
        self.name = name
        self.unit = unit
        self.CF = CF
        self.calibCheck = calibCheck

##%% ChannelObj properties
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self,newname):
        if isinstance(newname,str):
            self._name = newname
        else:
            raise TypeError('Channel name must be a string.')
            
    @property
    def unit(self):
        return self._unit
    
    @unit.setter
    def unit(self,newunit):
        if isinstance(newunit,str):
            if newunit == 'V':
                self._unit = newunit
                self.dBName = 'dBu'
                self.dBRef = 0.775
            elif newunit == 'Pa':
                self._unit = newunit
                self.dBName = 'dB(z)'
                self.dBRef = 2e-5
            elif newunit == 'W/m^2':
                self._unit = newunit
                self.dBName = 'dB'
                self.dBRef = 1e-12
            elif newunit == '':
                self._unit = ''
                self.dBName = 'dBFs'
                self.dBRef = 1
            else:
                raise TypeError(newunit+' unit not accepted. May be Pa, V or None.')
        else:
            raise TypeError('Channel unit must be a string.')
            
    @property
    def CF(self):
        return self._CF
    
    @CF.setter
    def CF(self,newCF):
        if isinstance(newCF,float) or isinstance(newCF,int):
            self._CF = newCF
        else:
            raise TypeError('Channel correction factor must be a number.')

    @property
    def calibCheck(self):
        return self._calibCheck
    
    @calibCheck.setter
    def calibCheck(self,newcalibCheck):
        if isinstance(newcalibCheck,bool):
            self._calibCheck = newcalibCheck
        else:
            raise TypeError('Channel calibration check must be True or False.')          
            
    
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
        
    Methods(args): meaning;

        - num_channels(): return the number of channels in the instace;
        - max_level(): return the channel's max levels;
        - play(): reproduce the timeSignal with default output device;
        - plot_time(): generates the signal's historic graphic;
        - plot_freq(): generates the signal's spectre graphic;
        - calib_voltage(refSignalObj,refVrms,refFreq): voltage calibration from an input SignalObj;
        - calib_pressure(refSignalObj,refPrms,refFreq): pressure calibration from an input SignalObj;
        - save_mat(filename): save a SignalObj to a .mat file;
    
    """
    
    def __init__(self,
                     signalArray=np.array([],ndmin=2).T,
                     domain='time',
                     *args,
                     **kwargs):
        # Checking input array dimensions
        if self.size_check(signalArray)>2:
            message = "No 'pyttaObj' is able handle arrays with more \
                        than 2 dimensions, '[:,:]', YET!."
            raise AttributeError(message)
        elif self.size_check(signalArray) == 1:
            signalArray = np.array(signalArray,ndmin=2).T
            
        super().__init__(*args,**kwargs)
        
        self.channels = []

        self.domain = domain or args[1]
        if self.domain == 'freq':
            self.freqSignal = signalArray # [-] signal in frequency domain
        elif self.domain == 'time':
            self.timeSignal = signalArray # [-] signal in time domain
        else:
            self.timeSignal = signalArray
            self.domain = 'time'
            print('Taking the input as a time domain signal')
        
        self._do_conform_channels()

        
##%% SignalObj Properties
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
    def timeSignal(self,newSignal): # when timeSignal have new ndarray value, calculate other properties
        if isinstance(newSignal,np.ndarray):            
            if self.size_check(newSignal) == 1:
                newSignal = np.array(newSignal,ndmin=2).T   
            self._timeSignal = np.array(newSignal)
            self._freqSignal = np.fft.rfft(self._timeSignal,axis=0,norm=None) # [-] signal in frequency domain
            self._numSamples = len(self._timeSignal) # [-] number of samples
            self._fftDegree = np.log2(self._numSamples) # [-] size parameter
            self._timeLength = self.numSamples / self.samplingRate # [s] signal time lenght
            self._timeVector = np.linspace(0,self.timeLength - (1/self.samplingRate),self.numSamples ) # [s] time vector (x axis)
            self._freqVector = np.linspace(0,(self.numSamples - 1) * self.samplingRate / (2*self.numSamples),
                                          (self.numSamples/2)+1 if self.numSamples%2==0 else (self.numSamples+1)/2 ) # [Hz] frequency vector (x axis)
            self._do_conform_channels()
        else: raise TypeError('Input array must be a numpy ndarray')
        return

    @property # when freqSignal is called returns the normalized ndarray
    def freqSignal(self): 
        normFreqSig = 1/len(self._freqSignal)*self._freqSignal
        return normFreqSig
    
    @freqSignal.setter
    def freqSignal(self,newSignal):
        if isinstance(newSignal,np.ndarray):            
            if self.size_check(newSignal) == 1:
                newSignal = np.array(newSignal,ndmin=2).T
            self._freqSignal = np.array(newSignal)
            self._timeSignal = np.fft.irfft(self._freqSignal,axis=0,norm=None)
            self._numSamples = len(self.timeSignal) # [-] number of samples
            self._fftDegree = np.log2(self.numSamples) # [-] size parameter
            self._timeLength = self.numSamples/self.samplingRate  # [s] signal time lenght
            self._timeVector = np.arange(0,self.timeLength, 1/self.samplingRate) # [s] time vector
            self._freqVector = np.linspace(0,(self.numSamples-1) * self.samplingRate / (2*self.numSamples),
                                           (self.numSamples/2)+1 if self.numSamples%2==0  else (self.numSamples+1)/2 ) # [Hz] frequency vector
            self._do_conform_channels()
        else: raise TypeError('Input array must be a numpy ndarray')
        return
                    
##%% SignalObj Methods
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
            maxAmplitude = np.max(np.abs(self.timeSignal[:,chIndex]))
            maxlvl.append(20*np.log10(maxAmplitude/self.channels[chIndex].dBRef))
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
        return
    
    def plot_time(self):
        """
        Time domain plotting method
        """
        # DB
        plot.figure( figsize=(10,5) )
        if self.num_channels() > 1:
            for chIndex in range(0,self.num_channels()):
                label = self.channels[chIndex].name+' ['+self.channels[chIndex].unit+']'
                plot.plot( self.timeVector,self.timeSignal[:,chIndex],label=label)
        else:
            chIndex = 0
            label = self.channels[chIndex].name+' ['+self.channels[chIndex].unit+']'
            plot.plot( self.timeVector, self.timeSignal[:,chIndex],label=label)            
        plot.legend(loc='best')
        plot.grid(color='gray', linestyle='-.', linewidth=0.4)
        plot.axis( ( self.timeVector[0] - 10/self.samplingRate, \
                    self.timeVector[-1] + 10/self.samplingRate, \
                    1.05*np.min( self.timeSignal ), \
                   1.05*np.max( self.timeSignal ) ) )
        plot.xlabel(r'$Time$ [s]')
        plot.ylabel(r'$Amplitude$')
        return
    
    def plot_freq(self,smooth=False):
        """
        Frequency domain plotting method
        """
        plot.figure( figsize=(10,5) )
        
        if self.num_channels() > 1:
            for chIndex in range(0,self.num_channels()):
                if smooth: 
                    Signal = signal.savgol_filter( np.squeeze(np.abs(self.freqSignal[:,chIndex])), 31, 3 ) 
                else: 
                    Signal = self.freqSignal[:,chIndex]
                dBSignal = 20 * np.log10( np.abs( Signal ) / self.channels[chIndex].dBRef )
                label = self.channels[chIndex].name+' ['+self.channels[chIndex].dBName+' ref.: '+str(self.channels[chIndex].dBRef)+' '+self.channels[chIndex].unit+']'
                plot.semilogx( self.freqVector,dBSignal,label=label)
        else:
            chIndex = 0
            if smooth: 
                Signal = signal.savgol_filter( np.squeeze(np.abs(self.freqSignal[:,chIndex])), 31, 3 ) 
            else: 
                Signal = self.freqSignal[:,chIndex]
            dBSignal = 20 * np.log10( np.abs( Signal ) / self.channels[chIndex].dBRef )
            label = self.channels[chIndex].name+' ['+self.channels[chIndex].dBName+' ref.: '+str(self.channels[chIndex].dBRef)+' '+self.channels[chIndex].unit+']'
            plot.semilogx( self.freqVector, dBSignal ,label=label)            
            
        plot.grid(color='gray', linestyle='-.', linewidth=0.4)        
        plot.legend(loc='best')
        if np.max(dBSignal) > 0:
            ylim = [1.05*np.min(dBSignal),1.12*np.max(dBSignal)]
        else:
            ylim = [np.min(dBSignal)-2,np.max(dBSignal)+2]
        plot.axis((self.freqMin,self.freqMax,ylim[0],ylim[1]))
        plot.xlabel(r'$Frequency$ [Hz]')
        plot.ylabel(r'$Magnitude$ in dB')
        return

    def calib_voltage(self,chIndex,refSignalObj,refVrms=1,refFreq=1000):
        """
        calibVoltage method: use informed SignalObj with a calibration voltage signal, and the reference RMS voltage to calculate the Correction Factor.
        
            >>> SignalObj.calibVoltage(chIndex,refSignalObj,refVrms,refFreq)
            
        argument: (default), (dtype), meaning;
            
            - chIndex: (), (int), channel index for calibration. Starts in 0;
            - refSignalObj: (), (SignalObj), SignalObj with the calibration recorded signal;
            - refVrms: (1.00), (float), the reference voltage provided by the voltage calibrator;
            - refFreq: (1000), (int), the reference sine frequency provided by the voltage calibrator;
            
        """
        if chIndex in range(self.num_channels()):
            Vrms = np.max(np.abs(refSignalObj.freqSignal[:,0]))
            print(Vrms)
            freqFound = np.round(refSignalObj.freqVector[np.where(np.abs(refSignalObj.freqSignal)==np.max(np.abs(refSignalObj.freqSignal)))[0]])
            if freqFound != refFreq:
                print('\x1b[0;30;43mATENTTION! Found calibration frequency ('+'%.2f'%freqFound+' [Hz]) differs from refFreq ('+'%.2f'%refFreq+' [Hz])\x1b[0m')
            self.channels[chIndex].CF = refVrms/Vrms
            self.channels[chIndex].unit = 'V'
            newtimeSignal = cp.deepcopy(self.timeSignal)
            newtimeSignal[:,chIndex] = self.timeSignal[:,chIndex]*self.channels[chIndex].CF
            self.timeSignal = newtimeSignal
            self.channels[chIndex].calibCheck = True
        else: raise IndexError('chIndex greater than channels number')
        return
        
    def calib_pressure(self,chIndex,refSignalObj,refPrms=1.00,refFreq=1000):
        """
        calibPressure method: use informed SignalObj, with a calibration acoustic pressure signal, and the reference RMS acoustic pressure to calculate the Correction Factor.
        
            >>> SignalObj.calibPressure(chIndex,refSignalObj,refPrms,refFreq)
            
        argument: (default), (dtype), meaning;
            
            - chIndex: (), (int), channel index for calibration. Starts in 0;
            - refSignalObj: (), (SignalObj), SignalObj with the calibration recorded signal;
            - refPrms: (1.00), (float), the reference pressure provided by the acoustic calibrator;
            - refFreq: (1000), (int), the reference sine frequency provided by the acoustic calibrator;
            
        """
        if chIndex in range(self.num_channels()):
            Prms = np.max(np.abs(refSignalObj.freqSignal[:,0]))
            print(Prms)
            freqFound = np.round(refSignalObj.freqVector[np.where(np.abs(refSignalObj.freqSignal)==np.max(np.abs(refSignalObj.freqSignal)))[0]])
            if freqFound != refFreq:
                print('\x1b[0;30;43mATENTTION! Found calibration frequency ('+'%.2f'%freqFound+' [Hz]) differs from refFreq ('+'%.2f'%refFreq+' [Hz])\x1b[0m')
            self.channels[chIndex].CF = refPrms/Prms
            self.channels[chIndex].unit = 'Pa'
            newtimeSignal = cp.deepcopy(self.timeSignal)
            newtimeSignal[:,chIndex] = self.timeSignal[:,chIndex]*self.channels[chIndex].CF
            self.timeSignal = newtimeSignal
            self.channels[chIndex].calibCheck = True
        else: raise IndexError('chIndex greater than channels number')
        return

    def save_mat(self,filename=time.ctime(time.time())):
        mySigObj = vars(self)            
        sio.savemat(filename,_to_dict(mySigObj),format='5')
    
    def _do_conform_channels(self):
        dCh = self.num_channels() - len(self.channels)
        if dCh > 0:
            for i in range(0,dCh): self.channels.append(ChannelObj(name='Channel '+str(i+self.num_channels())))
        if dCh < 0:
            for i in range(0,-dCh): self.channels.pop(-1)
            
    def __truediv__(self, other):
        """
        Frequency domain division method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike.")
        if other.samplingRate != self.samplingRate:
            raise TypeError("Both SignalObj must have the same sampling rate.")
        result = SignalObj(samplingRate=self.samplingRate)
        result._domain = 'freq'
        if self.size_check() > 1:
            if other.size_check() > 1:
                if other.size_check() != self.size_check():
                    raise ValueError("Both signal-like objects must have the same number of channels.")
                for channel in range(other.num_channels()):
                    result.freqSignal = self._freqSignal[:,channel] / other._freqSignal[:,channel]
            else:
                for channel in range(other.num_channels()):
                    result.freqSignal = self._freqSignal[:,channel] / other._freqSignal
        else: result.freqSignal = self._freqSignal / other._freqSignal
        return result
    
    
    def __add__(self, other):
        """
        Time domain addition method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike.")
        if other.samplingRate != self.samplingRate:
            raise TypeError("Both SignalObj must have the same sampling rate.")
        result = SignalObj(samplingRate=self.samplingRate)
        result.domain = 'time'
        if self.size_check() > 1:
            if other.size_check() > 1:
                if other.size_check() != self.size_check():
                    raise ValueError("Both signal-like objects must have the same number of channels.")
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:,channel] + other._timeSignal[:,channel]
            else:
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:,channel] + other._timeSignal
        else: result.timeSignal = self._timeSignal + other._timeSignal
        return result

    def __sub__(self, other):
        """
        Time domain subtraction method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike.")
        if other.samplingRate != self.samplingRate:
            raise TypeError("Both SignalObj must have the same sampling rate.")
        result = SignalObj(samplingRate=self.samplingRate)
        result.domain = 'time'
        if self.size_check() > 1:
            if other.size_check() > 1:
                if other.size_check() != self.size_check():
                    raise ValueError("Both signal-like objects must have the same number of channels.")
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:,channel] - other._timeSignal[:,channel]
            else:
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:,channel] - other._timeSignal
        else: result.timeSignal = self._timeSignal - other._timeSignal
        return result

    def _calc_spectrogram( self, timeData = None,  overlap=0.5, winType='hann', winSize=512 ):
        if timeData is None:
            timeData = self.timeSignal
            if self.size_check() > 1:
                timeData = timeData[:,0]
        window = eval('signal.windows.'+winType)(winSize)
        nextIdx = int(winSize*overlap)
        rng = int( timeData.shape[0]/winSize/overlap - 1 )
        _spectrogram = np.zeros(( winSize//2 + 1, rng ))
        for N in range(rng):
            try:
                strIdx = N*nextIdx
                endIdx = winSize + N*nextIdx
                sliceAudio = (8/np.sqrt(3))*window*timeData[ strIdx : endIdx ]
                sliceFFT = np.fft.rfft(sliceAudio, axis=0)
                sliceMag = np.absolute( sliceFFT )*(2/sliceFFT.size)
                _spectrogram[: , N] = sliceMag
            except IndexError:
                sliceAudio = timeData[ -winSize: ]
                sliceFFT = np.fft.rfft(sliceAudio, axis=0)
                sliceMag = np.absolute( sliceFFT )*(2/sliceFFT.size)
                _spectrogram[: , N] = sliceMag
        return _spectrogram


##%% ImpulsiveResponse class
class ImpulsiveResponse(PyTTaObj):
    def __init__(self, excitationSignal, recordedSignal,
                 coordinates={'points':[], 'reference':'south-west-floor corner', 'unit':'m'},
                 method='linear', winSize=None, overlap=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inputSignal = excitationSignal
        self._outputSignal = recordedSignal
        self._coordinates = coordinates
        self._methodInfo = {'method':method, 'winSize':winSize, 'overlap':overlap}
        self._irSignal = self._get_transferfunction(excitationSignal,
                                                   recordedSignal,
                                                   method=method,
                                                   winSize=winSize,
                                                   overlap=overlap)
        return

##%% Properties
    @property
    def inputSignal(self):
        return self._inputSignal

    @property
    def outputSignal(self):
        return self._outputSignal

    @property
    def irSignal(self):
        return self._irSignal
    
    @property
    def tfSignal(self):
        return self._irSignal

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def methodInfo(self):
        return self._methodInfo

##%% Private methods
    def _get_transferfunction(self, inputSignal, outputSignal, method='linear',
                             winSize=None, overlap=None):
        
        if type(inputSignal) != type(outputSignal):
            raise TypeError("Only signal-like objects can become and Impulsive Response.")
        elif inputSignal.samplingRate != outputSignal.samplingRate:
            raise ValueError("Both signal-like objects must have the same sampling rate.")

        if method == 'linear':
            result = outputSignal / inputSignal
            
        elif method == 'H1':
            if winSize is None: winSize = inputSignal.samplingRate//2
            if overlap is None: overlap = 0.5
            result = SignalObj(np.zeros((winSize//2 + 1, outputSignal.freqSignal.shape[1])),
                               domain='freq',samplingRate=inputSignal.samplingRate)
            if outputSignal.num_channels() > 1:
                if inputSignal.num_channels() > 1:
                    if inputSignal.num_channels() != outputSignal.num_channels():
                        raise ValueError("Both signal-like objects must have the same number of channels.")
                    for channel in range(outputSignal.num_channels()):
                        XYXX = self._calc_csd_tf(inputSignal.timeSignal[:,channel],
                                                outputSignal.timeSignal[:,channel],
                                                inputSignal.samplingRate,
                                                winSize, winSize*overlap)
                        result.freqSignal[:,channel] = XYXX
                else:
                    for channel in range(outputSignal.num_channels()):
                        XYXX = self._calc_csd_tf(inputSignal.timeSignal,
                                                outputSignal.timeSignal[:,channel],
                                                inputSignal.samplingRate,
                                                winSize, winSize*overlap)
                        result.freqSignal[:,channel] = XYXX
            else:
                XYXX = self._calc_csd_tf(inputSignal.timeSignal,
                                        outputSignal.timeSignal,
                                        inputSignal.samplingRate,
                                        winSize, winSize*overlap)
                result.freqSignal = XYXX
        
        elif method == 'H2':
            if winSize is None: winSize = inputSignal.samplingRate//2
            if overlap is None: overlap = 0.5
            result = SignalObj(samplingRate=inputSignal.samplingRate)
            result.domain='freq'
            if outputSignal.num_channels() > 1:
                if inputSignal.num_channels() > 1:
                    if inputSignal.num_channels() != outputSignal.num_channels():
                        raise ValueError("Both signal-like objects must have the same number of channels.")
                    for channel in range(outputSignal.num_channels()):
                        YXYY = self._calc_csd_tf(outputSignal.timeSignal[:,channel],
                                                 inputSignal.timeSignal[:,channel],
                                                 inputSignal.samplingRate,
                                                 winSize, winSize*overlap)
                        result.freqSignal[:,channel] = 1/YXYY
                else:
                    YXYY = self._calc_csd_tf(outputSignal.timeSignal[:,channel],
                                             inputSignal.timeSignal,
                                             inputSignal.samplingRate,
                                             winSize, winSize*overlap)
                    result.freqSignal[:,channel] = 1/YXYY
            else:
                YXYY = self._calc_csd_tf(outputSignal.timeSignal,
                                         inputSignal.timeSignal,
                                         inputSignal.samplingRate,
                                         winSize, winSize*overlap)
                result.freqSignal = 1/YXYY
        elif method == 'Ht':
            if winSize is None: winSize = inputSignal.samplingRate//2
            if overlap is None: overlap = 1/2
            result = SignalObj(samplingRate=inputSignal.samplingRate, domain='freq')
            if outputSignal.size_check() > 1:
                if inputSignal.size_check() > 1:
                    if inputSignal.num_channels() != outputSignal.num_channels():
                        raise ValueError("Both signal-like objects must have the same number of channels.")
                else:
                    pass
            else:
                pass
            pass
        return result    # end of function get_transferfunction() 


    def _calc_csd_tf(self, sig1, sig2, samplingRate,
                     numberOfSamples, overlapSamples):
        
        f, S11 = signal.csd(sig1, sig1, samplingRate,
                            nperseg = numberOfSamples, 
                            noverlap = overlapSamples, axis=0)
        
        f, S12 = signal.csd(sig1, sig2, samplingRate,
                            nperseg = numberOfSamples,
                            noverlap = overlapSamples, axis=0)
        return S12/S11

    def _coord_points_per_channel(self):
        pass # TODO
    
##%% Measurement class
class Measurement(PyTTaObj):
    """
    Measurement object class created to define some properties and methods to
    be used by the playback, recording and processing classes. It is a private
    class
    
    Properties(self): (default), (dtype), meaning;
    
        - device: (system default), (list/int), list of input and output devices;
        - inChannel: ([1]), (list/int), list of device's input channel used for recording;
        - outChannel: ([1]), (list/int), list of device's output channel used for playing/reproducing a signalObj;

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
                 channelName=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args,**kwargs)
        self.device = device # device number. For device list use sounddevice.query_devices()
        self.inChannel = inChannel # input channels
        self.outChannel = outChannel # output channels
        self.channelName = channelName
        return

##%% Measurement Properties

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self,newDevice):
        self._device = newDevice
        return
    
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
        return
            
    @property
    def outChannel(self):
        return self._outChannel

    @outChannel.setter
    def outChannel(self,newOutputChannel):
        self._outChannel = newOutputChannel
            
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
        return

##%% RecMeasure class        
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
        return

##%% Rec Properties
    @property
    def timeLength(self):
        return self._timeLength
    
    @timeLength.setter
    def timeLength(self,newLength):
        self._timeLength = np.round( newLength, 2 )
        self._numSamples = self.timeLength * self.samplingRate
        self._fftDegree = np.round( np.log2( self.numSamples ), 2 )
        return
        
    @property
    def fftDegree(self):
        return self._fftDegree
    
    @fftDegree.setter
    def fftDegree(self,newDegree):
        self._fftDegree = np.round( newDegree, 2 )
        self._numSamples = 2**self.fftDegree
        self._timeLength = np.round( self.numSamples / self.samplingRate, 2 )
        return

##%% Rec Methods
    def run(self):
        """
        Run method: starts recording during Tmax seconds
        Outputs a signalObj with the recording content
        """
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
        for chIndex in range(self.recording.num_channels()):
            self.recording.channels[chIndex].name = self.channelName[chIndex]
        self.recording.timeStamp = time.ctime(time.time())
        self.recording.freqMin, self.recording.freqMax = (self.freqMin,self.freqMax)
        self.recording.comment = 'SignalObj from a Rec measurement'
        _print_max_level(self.recording,kind='input')
        return self.recording    
    
##%% PlayRecMeasure class 
class PlayRecMeasure(Measurement):
    """
    Playback and Record object

    Properties(self), (default), (dtype), meaning:
        
		- excitation: (SignalObj), (SignalObj), signal information used to reproduce (playback);

    Properties(inherited): 	(default), (dtype), meaning;
    
        - device: (system default), (list/int), list of input and output devices;
        - inChannel: ([1]), (list/int), list of device's input channel used for recording;
        - outChannel: ([1]), (list/int), list of device's output channel used for playing/reproducing a signalObj;
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
        return

##%% PlayRec Methods
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
                             dtype = 'float64'
                             ) # y_all(t) - out signal: x(t) conv h(t)
        recording = np.squeeze( recording ) # turn column array into line array
        self.recording = SignalObj(signalArray=recording,domain='time',samplingRate=self.samplingRate )
        for chIndex in range(self.recording.num_channels()):
            self.recording.channels[chIndex].name = self.channelName[chIndex]
        self.recording.timeStamp = timeStamp
        self.recording.freqMin, self.recording.freqMax = (self.freqMin,self.freqMax)
        self.recording.comment = 'SignalObj from a PlayRec measurement'
        _print_max_level(self.excitation,kind='output')
        _print_max_level(self.recording,kind='input')
        return self.recording

##%% PlayRec Properties
    @property
    def excitation(self):
        return self._excitation        
    @excitation.setter
    def excitation(self,newSignalObj):
        self._excitation = newSignalObj
        return
        
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


##%% FRFMeasure class     
class FRFMeasure(PlayRecMeasure):
    """
    Transferfunction object
        
    Properties(inherited): 	(default), (dtype), meaning;
    
        - excitation: (SignalObj), (SignalObj), signal information used to reproduce (playback);
        - device: (system default), (list/int), list of input and output devices;
        - inChannel: ([1]), (list/int), list of device's input channel used for recording;
        - outChannel: ([1]), (list/int), list of device's output channel used for playing/reproducing a signalObj;
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
        return
        
    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Divides the recorded signalObj by the excitation signalObj to generate a transferfunction
        Outputs the transferfunction signalObj
        """
        self.recording = super().run()
        self.transferfunction = self.recording/self.excitation
        self.transferfunction.timeStamp = self.recording.timeStamp
        self.transferfunction.freqMin, self.recording.freqMax = (self.freqMin,self.freqMax)
        self.recording.comment = 'SignalObj from a FRF measurement'
        return self.transferfunction
    
##%% Sub functions
def _print_max_level(sigObj,kind):
    if kind == 'output':
        for chIndex in range(sigObj.num_channels()):
            print('max output level (excitation) on channel ['+str(chIndex+1)+']: '+'%.2f'%sigObj.max_level()[chIndex]+' '+sigObj.channels[chIndex].dBName+' - ref.: '+str(sigObj.channels[chIndex].dBRef)+' ['+sigObj.channels[chIndex].unit+']')
            if sigObj.max_level()[chIndex] >= 0:
                print('\x1b[0;30;43mATENTTION! CLIPPING OCCURRED\x1b[0m')
    if kind == 'input':
        for chIndex in range(sigObj.num_channels()):
            print('max input level (recording) on channel ['+str(chIndex+1)+']: '+'%.2f'%sigObj.max_level()[chIndex]+' '+sigObj.channels[chIndex].dBName+' - ref.: '+str(sigObj.channels[chIndex].dBRef)+' ['+sigObj.channels[chIndex].unit+']')
            if sigObj.max_level()[chIndex] >= 0:
                print('\x1b[0;30;43mATENTTION! CLIPPING OCCURRED\x1b[0m')
        return

def _to_dict(thing):
    
    # From SignalObj to dict
    if isinstance(thing,SignalObj):
        mySigObj = vars(thing)
        dictime = {}
        for key, value in mySigObj.items():
            # Recursive stuff for values
            dictime[key] = _to_dict(value)
        # Recursive stuff for resultant dict
        return _to_dict(dictime)
    
    # From ChannelObj to dict
    elif isinstance(thing,ChannelObj):
        myChObj = vars(thing)
        dictime = {}
        for key, value in myChObj.items():
            dictime[key] = _to_dict(value)
        # Recursive stuff for resultant dict
        return _to_dict(dictime)
    
   # From a bad dict to a good dict
    elif isinstance(thing,dict):
        dictime = {}
        for key, value in thing.items():
            # Removing spaces from dict keys
            if key.find(' ') >= 0:
                key = key.replace(' ','')
            # Removing underscores from dict keys
            if key.find('_') >= 0:
                key = key.replace('_','')
            # Removing empty dicts from values
            if isinstance(value,dict) and len(value) == 0:
                dictime[key] = 0
            # Removing None from values
            if value is None:
                dictime[key] = 0
            # Recursive stuff
            dictime[key] = _to_dict(value)
        return dictime
    
    # Turning lists into dicts with 'T + listIndex' keys
    elif isinstance(thing,list):
        dictime = {}
        j = 0
        for item in thing:
            dictime['T'+str(j)] = _to_dict(item)
            j=j+1
        return dictime
   
    elif thing is None:
        return 0
   
    else:
        return thing
