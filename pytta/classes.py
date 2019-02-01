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
    >>> pytta.generate.measurement('rec', domain = 'time', timeLen = 5)
    
This way, the default settings will be loaded into any object instantiated.

User intended classes:
    
    >>> pytta.signalObj()
    >>> pytta.RecMeasure()
    >>> pytta.PlayRecMeasure()
    >>> pytta.FRFMeasure()
    
For further information see the specific class, or method, documentation
"""
#%% Importing modules
#import pytta as pa
from .properties import Default
import numpy as np
import matplotlib.pyplot as plot
import scipy.signal as signal
import sounddevice as sd


class pyttaObj(object):
    """
    PyTTa object class to define some properties and methods to be used 
    by any signal and processing classes. pyttaObj is a private class created
    just to shorten attributes declaration to each PyTTa class.
    
    Properties(self):    (default),     meaning
        - Fs:            (44100),       signal's sampling rate
        - Flim:          ([20,20000]),  frequency bandwidth limits
        - comment:       (' '),         some commentary about the signal or measurement object
        
    """
    
    def __init__(self,Fs=1,
                 Finf = 17,
                 Fsup = 22050,
                 comment=Default().comment
                 ):
        self.Fs = Fs
        self.Flim = [Finf, Fsup]
        self.comment = comment
        
    def __call__(self):
        for name, value in vars(self).items():
            if len(name)<=7:
                print(name+'\t\t =',value)
            else: 
                print(name+'\t =',value)

    

class signalObj(pyttaObj):
	"""
    Signal object class.
    
    Properties(self): 	   	(default),   	meaning  
        - timeSignal:   	(ndarray),   	signal at time domain;
        - timeVector:   	(ndarray),   	time reference vector for timeSignal;
        - freqSignal:   	(ndarray),   	signal at frequency domain;
        - freqVector:   	(ndarray),   	frequency reference vector for freqSignal;
        - N:            	(samples),   	signal's number of samples;
        - timeLen:      	(seconds),   	signal's duration;
        
    Properties(inherited):  (default),   	meaning
        - Fs: 	    	  	(44100), 	 	signal's sampling rate;
        - Flim: 	 	 	([20,20000]), 	frequency bandwidth limits;
        - comment: 	 	 	(' ') 	 	 	some commentary about the signal;        
        
    Methods: 	 	 	 	meaning
        - play():  	 	 	reproduce the timeSignal using the default audio output device;
        - plot_time():  	generates the signal's historic graphic;
        - plot_freq():  	generates the signal's spectre graphic;
    
    """
	def __init__(self,arr_in=np.array([0]),domain=None,*args,**kwargs):
		if self.sizeCheck(arr_in)>2:
			message = "No 'pyttaObj' is able handle arrays with more than 2 dimensions, '[:,:]', YET!."
			raise AttributeError(message)
		else:
			pass
		super().__init__(*args,**kwargs)
		self.domain = domain
		if self.domain == 'freq':
			self.freqSignal = arr_in # [-] signal in frequency domain
		elif self.domain == 'time':
			self.timeSignal = arr_in # [-] signal in time domain
		else:
			self.timeSignal = arr_in
    
	@property # when t is called as object.t it returns the ndarray
	def timeSignal(self):
		return self._timeSignal
	@timeSignal.setter
	def timeSignal(self,newt): # when t have new ndarray value, calculate other properties
		self._timeSignal = np.array(newt)
		self.N = len(self.timeSignal) # [-] number of samples
		self.timeLen = self.N/self.Fs # [s] signal time lenght
		self.timeVector = np.linspace(0, self.timeLen-(1/self.Fs), self.N) # [s] time vector
#		if len(self.timeVector) or len(self.timeSignal) != self.N:
#			self.timeVector = self.timeVector[:self.N]
#			self.timeSignal = self.timeSignal[:self.N] 
		self.freqVector = np.linspace(0, (self.N-1)*self.Fs/self.N, self.N)# [Hz] frequency vector
		self._freqSignal = (2/self.N)*np.transpose( np.fft.fft( self.timeSignal.transpose() ) ) # [-] signal in frequency domain
        
	@property
	def freqSignal(self):
		return self._freqSignal
	@freqSignal.setter
	def freqSignal(self,newjw):
		self._freqSignal = np.array(newjw)
		self._timeSignal = np.transpose( np.real(np.fft.ifft(self.freqSignal.transpose())) )
		self.N = len(self.timeSignal) # [-] number of samples
		self.timeLen = self.N/self.Fs # [s] signal time lenght
		self.timeVector = np.arange(0, self.timeLen, 1/self.Fs) # [s] time vector
#		self.timeVector = self.timeVector[0:len(self.timeSignal)]
		self.freqVector = np.linspace(0, (self.N-1)*self.Fs/self.N, self.N)# [Hz] frequency vector

	def __truediv__(self, other):
		"""
		Frequency domain division method
		"""
		resul = signalObj(Fs=self.Fs)
		resul.freqSignal = self.freqSignal/other.freqSignal
		return resul
    
#	def __mul__(self, other):
#		"""
#		Frequency domain multiplication method
#		"""
#		resul = signalObj(Fs=self.Fs)
#		resul.freqSignal = self.freqSignal*other.freqSignal
#		return resul

	def __add__(self, other):
		"""
		Time domain addition method
		"""
		resul = signalObj(Fs=self.Fs)
		resul.timeSignal = self.timeSignal+other.timeSignal
		return resul

#	def __sub__(self, other):
#		"""
#		Time domain subtraction method
#		"""
#		resul = signalObj(Fs=self.Fs)
#		resul.timeSignal = self.timeSignal-other.timeSignal
#		return resul
	
	def mean(self):
		return signalObj(np.mean(self.timeSignal,1),"time",self.Fs)
	
#	def numChannels(self):
#		sz = self.sizeCheck()
#		return np.shape(self.timeSignal)
		
	def sizeCheck(self,inputArray = []):
		if inputArray == []: inputArray = self.timeSignal
		return np.size(np.shape(inputArray))


	def play(self,outch=None,latency='low'):
		"""
		Play method
		"""
		if outch == None:
			try:
				numChannels = np.shape(self.timeSignal)[1]
			except IndexError:
				numChannels = 1
			if numChannels <=1:
				outch = Default().outChannel
			elif numChannels > 1:
				outch = np.arange(1,numChannels+1)

		sd.play(self.timeSignal,self.Fs,mapping=outch)
			   
#   def plot(self): # TODO
#        ...

	def plot_time(self):
		"""
		Time domain plotting method
		"""
		plot.figure(figsize=(10,5))
		plot.plot(self.timeVector,self.timeSignal)
		plot.axis([self.timeVector[0] - 10/self.Fs, self.timeVector[-1] + 10/self.Fs, 1.05*np.min(self.timeSignal), 1.05*np.max(self.timeSignal)])
		plot.xlabel(r'$Time$ [s]')
		plot.ylabel(r'$Amplitude$ [-]')


	def plot_freq(self):
		"""
		Frequency domain plotting method
		"""
		plot.figure(figsize=(10,5))
		Hjw_smooth = signal.savgol_filter(abs(self.freqSignal.transpose()),31,3);
		dB_smooth = 20*np.log10(np.abs(Hjw_smooth))
		plot.semilogx(self.freqVector,dB_smooth.transpose())
#		dB_self = 20*np.log10(np.abs(self.freqSignal))
#		plot.semilogx(self.freqVector,dB_self )
		plot.axis((15, 22050, 1.05*np.min(dB_smooth), 1.05*np.max(dB_smooth)))
		plot.xlabel(r'$Frequency$ [Hz]')
		plot.ylabel(r'$Magnitude$ [dBFS]')

#        plot.legend()
        

class Measurement(pyttaObj):
    """
    Measurement object class created to define some properties and methods to
    be used by the playback, recording and processing classes. It is a private
    class
    
    Properties(self): 	 	(default), 	 	 	meaning
        - device: 	 	 	(system default),  	list of input and output devices;
        - inch: 	 	 	([1]), 	 	 	 	list of device's input channel used for recording;
        - outch: 	 	 	([1]), 	 	 	 	list of device's output channel used for playing/reproducing a signalObj

    Properties(inherited): 	(default), 	 	 	meaning
        - Fs: 	 	 	 	(44100), 	 	 	measurement's sampling rate;
        - Flim: 	 	 	([20,20000]), 	 	frequency bandwidth limits;
        - comment: 	 	 	(' '), 	 	 	 	some commentary about the signal;        
        
    """
    def __init__(self,
                 device=None,
                 inch=None,
                 outch=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args,**kwargs)
        self.device = device # device number. For device list use sounddevice.query_devices()
        self.inch = inch # input channels
        self.outch = outch # output channels
        
     
class RecMeasure(Measurement):
    """
    Signal Recording object
    
    Properties(self) 	 	 (default), 	 	meaning:
		- domain:  	 	 	 ('samples'), 	 	Information about the recording length. May be 'time' or 'samples';
		- fftDeg: 	 	 	 (18),  	 	 	number of samples will be 2**fftDeg. Used if domain is set to 'samples';
		- timeLen: 	 	 	 (10), 	 	 	 	time length of the recording. Used if domain is set to 'time';

    Properties(inherited) 	(default), 	 	 	meaning:
        - device: 	 	 	(system default),  	list of input and output devices;
        - inch: 	 	 	([1]), 	 	 	 	list of device's input channel used for recording;
        - outch: 	 	 	([1]), 	 	 	 	list of device's output channel used for playing/reproducing a signalObj
        - Fs: 	 	 	 	(44100), 	 	 	recording's sampling rate;
        - Flim: 	 	 	([20,20000]), 	 	frequency bandwidth limits;
        - comment: 	 	 	(' '), 	 	 	 	some commentary about the signal;        

	Methods  	 	meaning:
		- run(): 	starts recording using the inch and device information, during timeLen seconds;
		

    """
    def __init__(self,domain=None,
                 fftDeg=None,
                 timeLen=None,
                 *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.domain = domain
        if self.domain == 'samples':
            self.fftDeg = fftDeg
        elif self.domain == 'time':
            self.timeLen = timeLen
        else:
            self._timeLen = None
            self._fftDeg=None
    
    @property
    def timeLen(self):
        return self._timeLen
    @timeLen.setter
    def timeLen(self,newT):
        self._timeLen = np.round(newT,2)
        self.N = self._timeLen*self.Fs
        self._fftDeg = np.round(np.log2(self.N),2)
        
    @property
    def fftDeg(self):
        return self._fftDeg
    @fftDeg.setter
    def fftDeg(self,newDeg):
        self._fftDeg = np.round(newDeg,2)
        self.N = 2**self._fftDeg
        self._timeLen = np.round(self.N/self.Fs,2)
                
    def run(self):
        """
        Run method: starts recording during Tmax seconds
        Outputs a signalObj with the recording content
        """
        self.recording = sd.rec(self.N,
                                self.Fs,
                                mapping = self.inch,
                                blocking=True,
                                latency='low',
                                dtype = 'float32'
                                )
        self.recording = np.squeeze(self.recording)
        self.recording = signalObj(self.recording,'time',self.Fs)
#        maxOut = max(abs(self.recording.timeSignal[:,:]))
#        print('max input level (recording): ', 20*np.log10(maxOut), 'dBFs - ref.: 1 [-]')
        return self.recording
    
    
class PlayRecMeasure(Measurement):
    """
    Playback and Record object

    Properties(self) 	 	 (default), 	 	meaning:
		- excitation:  	 	 (signalObj), 	 	Signal information used to reproduce (playback);
		- Fs:                (44100), 	 	 	signal's sampling rate;
		- Flim: 	 	 	 ([20,20000]), 	 	frequency bandwidth limits;
		- N:    	 	 	 (len(timeSignal)), number of samples will be 2**fftDeg. Used if domain is set to 'samples';
		- timeLen: 	 	 	 (N/Fs), 	 	 	time length of the recording. Used if domain is set to 'time';

    Properties(inherited): 	(default), 	 	 	meaning:
        - device: 	 	 	(system default),  	list of input and output devices;
        - inch: 	 	 	([1]), 	 	 	 	list of device's input channel used for recording;
        - outch: 	 	 	([1]), 	 	 	 	list of device's output channel used for playing/reproducing a signalObj
        - comment: 	 	 	(' '), 	 	 	 	some commentary about the signal;        

	Methods 	  	 	meaning:
		- run(): 	 	starts playing the excitation signal and recording during the excitation timeLen duration;

    """
    def __init__(self,excitation=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if excitation is None:
            self._excitation = None
        else:
            self.excitation = excitation
            
    @property
    def excitation(self):
        return self._excitation        
    @excitation.setter
    def excitation(self,newex):
        self._excitation = newex
        self.Fs = self._excitation.Fs
        self.N = self._excitation.N
        self.timeLen = self._excitation.timeLen # [s] measurement time
        self.Flim = self._excitation.Flim
        
    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Outputs a signalObj with the recording content
        """
        self.recording = sd.playrec(self.excitation.timeSignal,
                             samplerate=self.Fs, 
                             input_mapping=self.inch,
                             output_mapping=self.outch,
                             device=self.device,
                             blocking=True,
                             latency='low',
                             dtype = 'float32'
                             ) # y_all(t) - out signal: x(t) conv h(t)
        self.recording = np.squeeze(self.recording) # turn column array into line array
        self.recording = signalObj(self.recording,'time',self.Fs)
#        print('max output level (excitation): ', 20*np.log10(max(self.excitation.timeSignal)), 'dBFs - ref.: 1 [-]')
#        print('max input level (recording): ', 20*np.log10(max(self.recording.timeSignal)), 'dBFs - ref.: 1 [-]')
        return self.recording
   
     
class FRFMeasure(PlayRecMeasure):
    """
    Transferfunction object

    Properties(self)   (default), 	 	 	meaning:
		- excitation:  (signalObj), 	 	Signal information used to reproduce (playback);
		- Fs:          (44100), 	 	 	signal's sampling rate;
		- Flim: 	   ([20,20000]), 	 	frequency bandwidth limits;
		- N:    	   (len(timeSignal)), 	number of samples;
		- timeLen: 	   (N/Fs), 	 	 	 	time length of the recording.


    Properties(inherited) 	(default), 	 	 	meaning:
        - device: 	 	 	(system default),  	list of input and output devices;
        - inch: 	 	 	([1]), 	 	 	 	list of device's input channel used for recording;
        - outch: 	 	 	([1]), 	 	 	 	list of device's output channel used for playing/reproducing a signalObj
        - comment: 	 	 	(' '), 	 	 	 	some commentary about the signal.
		
		
	Methods 	  	 	meaning:
		- run(): 	 	starts playing the excitation signal and recording during the excitation timeLen duration. At the end of recording calculates the transferfunction between recorded and reproduced signals;

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
        return self.transferfunction