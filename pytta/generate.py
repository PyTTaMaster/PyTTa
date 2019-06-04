# -*- coding: utf-8 -*-
"""
Generate
=========
  
@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
- Matheus Lazarin Alberto, mtslazarin@gmail.com

    This submodule provides the tools for instantiating the measurement and
    signal objects to be used. We strongly recommend the use of this submodule
    instead of directly instantiating classes, except when necessary.
    
    The signal generating functions already have set up a few good practices
    on signal generation and reproduction through audio IO interfaces, like
    silences at beginning and ending of the signal, as well as fade ins and fade
    out to asvoid abrupt audio currents from flowing and causing undesired peaks
    at start/ending of reproduction.
    
    On the measurement side, it tries to set up the environment by already giving
    excitation signals, or 

    User intended functions:
        
        >>> pytta.generate.sin()
        >>> pytta.generate.sweep()
        >>> pytta.generate.noise()
        >>> pytta.generate.impulse()
        >>> pytta.generate.measurement()
    
    For further information see the specific function documentation
"""

##%% Import modules
from .classes import SignalObj, RecMeasure, FRFMeasure, PlayRecMeasure
from pytta import default
from scipy import signal
import numpy as np


def sin(Arms = 0.5,
        freq = 1000,
        timeLength = 1,
        phase = 2*np.pi,
        samplingRate = default.samplingRate,
        fftDegree = None):
    """
    Generates a sine signal with the traditional parameters plus some PyTTa
    options.
    
    Creation parameters:
    --------------------    
        * Arms (float) (optional):
            The signal's RMS amplitude. 
            
            >>> Apeak = Arms*sqrt(2);
        
        * freq (float) (optional):
            Nothing to say;
        
        * timeLength (float) (optional):
            Sine timeLength in seconds;
            
        * fftDegree (int) (optional);
            2**fftDegree signal's number of samples;
            
        * phase (float) (optional):
            Sine phase in radians;
            
        * samplingRate (int) (optional):
            Nothing to say;
            
    """
    if fftDegree != None:
        timeLength = 2**(fftDegree)/samplingRate
    t = np.linspace(0,timeLength - (1/samplingRate),samplingRate*timeLength)
    sin = Arms*(2**(1/2)) * np.sin(2*np.pi*freq*t+phase)
    sinSigObj = SignalObj(sin,domain='time',samplingRate=samplingRate)
    return sinSigObj
    

def sweep(freqMin = None,
          freqMax = None,
          samplingRate = None,
          fftDegree = None,
          startMargin = None,
          stopMargin = None,
          method = 'logarithmic',
          windowing = 'hann'):
    """
   Generates a chirp signal defined by the "method" input, windowed, with
   silence interval at the beggining and end of the signal, plus a hanning
   fade in and fade out.
 
   >>> x = pytta.generate.sweep()
   >>> x.plot_time()

   Return a signalObj containing a logarithmic chirp signal from 17.8 Hz
   to 22050 Hz, with a fade in beginning at 17.8 Hz time instant and ending at
   the 20 Hz time instant; plus a fade out beginning at 20000 Hz time instant
   and ending at 22050 Hz time instant.

   The fade in and the fade out are made with half hanning window. First half
   for the fade in and last half for the fade out. Different number of points
   are used for each fade, so the number of time samples during each frequency
   is respected.

    """
    if freqMin is None: freqMin = default.freqMin
    if freqMax is None: freqMax = default.freqMax
    if samplingRate is None: samplingRate = default.samplingRate
    if fftDegree is None: fftDegree = default.fftDegree
    if startMargin is None: startMargin = default.startMargin
    if stopMargin is None: stopMargin = default.stopMargin
    
    freqLimits = np.array( [ freqMin / ( 2**(1/6) ), \
                           min( freqMax*( 2**(1/6) ), \
                                   samplingRate/2 )
                           ] ) # frequency limits [Hz]
    samplingTime = 1/samplingRate # [s] sampling period
    
    stopSamples = stopMargin*samplingRate 
    # [samples] initial silence number of samples
    
    startSamples = startMargin*samplingRate
    # [samples] ending silence number of samples
    
    marginSamples = startSamples + stopSamples
    # [samples] total silence number of samples
    
    numSamples = 2**fftDegree # [samples] full signal number of samples
    
    sweepSamples = numSamples - marginSamples +1 # [samples] actual sweep number of samples
    
    if sweepSamples < samplingRate/10:
        raise Exception('Too small resultant sweep. For such big margins you must increase your fftDegree.')
    
    sweepTime = sweepSamples/samplingRate # [s] sweep's time length
    timeVecSweep = np.arange(0, sweepTime, samplingTime) # [s] sweep time vector
    if timeVecSweep.size > sweepSamples:
        timeVecSweep = timeVecSweep[0:int(sweepSamples)] # adjust length
    sweep = 0.8*signal.chirp(timeVecSweep, \
                             freqLimits[0],\
                             sweepTime, \
                             freqLimits[1],\
                              'logarithmic', \
                              phi=-90) # sweep, time domain
    sweep = __do_sweep_windowing(sweep, \
                                 timeVecSweep, \
                                 freqLimits, \
                                 freqMin, \
                                 freqMax, \
                                 windowing) # fade in and fade out
    timeSignal = np.concatenate( ( np.zeros( \
                                   int(startSamples) ), \
                                   sweep, \
                                   np.zeros( \
                                   int(stopSamples) 
                                   ) ) ) # add initial and ending sileces
    if timeSignal.size != numSamples:
        timeSignal = timeSignal[0:int(numSamples)] # adjust length
    
    sweepSignal = SignalObj(signalArray=timeSignal,domain='time',samplingRate=samplingRate) 
    # transforms into a pytta signalObj
    
    sweepSignal._freqMin, sweepSignal._freqMax \
            = freqLimits[0], freqLimits[1] 
    # pass on the frequency limits considering the fade in and fade out
    return sweepSignal

def __do_sweep_windowing(inputSweep,
                        timeVecSweep,
                        freqLimits,
                        freqMin,
                        freqMax,
                        window):
    """
    Applies a fade in and fade out that are minimum at the chirp start and end,
    and maximum between the time intervals corresponding to Finf and Fsup.
    
    """
    
    freqSweep = freqLimits[0]*( ( freqLimits[1] \
                          / freqLimits[0] ) \
                        **( 1/max(timeVecSweep) ) ) \
                        **timeVecSweep # frequencies at time instants: freq(t)
    freqMinSample = np.where(freqSweep<=freqMin) # exact sample where the chirp
                                              # reaches freqMin [Hz]
    freqMinSample = freqMinSample[-1][-1]

    freqMaxSample = np.where(freqSweep<=freqMax) # exact sample where the chirp
                                                # reaches freqMax [Hz]
    freqMaxSample = len(freqSweep) - freqMaxSample[-1][-1]
    windowStart = signal.hann(2*freqMinSample)
    windowEnd = signal.hann(2*freqMaxSample)
    
    # Uses first half of windowStart, last half of windowEnd, and a vector of 
    # ones with the remaining length, in between the half windows
    fullWindow = np.concatenate((windowStart[0:freqMinSample], \
                          np.ones( int( len(freqSweep) \
                                       - freqMinSample \
                                       - freqMaxSample \
                                       + 1) ), \
                          windowEnd[freqMaxSample:-1] ) )
    newSweep = fullWindow * inputSweep
    return newSweep
 
    
 
def noise(kind = 'white',
          samplingRate = None,
          fftDegree = None,
          startMargin = None,
          stopMargin = None,
          windowing = 'hann'
          ):
    """Generates a noise of kind White, Pink (TO DO) or Blue (TO DO), with a silence at the
	begining and ending of the signal, plus a fade in to avoid abrupt speaker
	excursioning. All noises have normalized amplitude.
	
		White noise is generated using numpy.randn between [[1];[-1]];
	
		Pink noise is still in progress;

        Blue noise is still in progress;
    """
    
    if samplingRate is None: samplingRate = default.samplingRate
    if fftDegree is None: fftDegree = default.fftDegree
    if startMargin is None: startMargin = default.startMargin
    if stopMargin is None: stopMargin = default.stopMargin


    stopSamples = stopMargin*samplingRate 
    # [samples] initial silence number of samples
    
    startSamples = startMargin*samplingRate
    # [samples] ending silence number of samples
    
    marginSamples = startSamples + stopSamples
    # [samples] total silence number of samples
    
    numSamples = 2**fftDegree # [samples] full signal number of samples
    noiseSamples = int(numSamples - marginSamples) # [samples] Actual noise number of samples
    if kind.upper() in ['WHITE','FLAT']:
        noiseSignal = np.random.randn(noiseSamples)
#	elif kind.upper() == 'PINK':                            # TODO
#		noiseSignal = np.randn(Nnoise)
#		noiseSignal = noiseSignal/max(abs(noiseSignal))
#		noiseSignal = __do_pink_filtering(noiseSignal)
#	elif kind.upper() == 'BLUE':                            # TODO
#		noiseSignal = np.randn(Nnoise)
#		noiseSignal = noiseSignal/max(abs(noiseSignal))
#		noiseSignal = __do_blue_filtering(noiseSignal)

    noiseSignal = __do_noise_windowing( noiseSignal, noiseSamples, windowing )
    noiseSignal = noiseSignal / max( abs( noiseSignal ) )
    fullSignal = np.concatenate( ( np.zeros( int(startSamples) ), \
                              noiseSignal, \
                              np.zeros( int(stopSamples) ) ) )
    fullSignal = SignalObj(signalArray=fullSignal,domain='time',samplingRate=samplingRate)
    return fullSignal

def __do_noise_windowing(inputNoise,
                        noiseSamples,
                        window):
	
    fivePercentSample = int( (5/100) * (noiseSamples) ) # sample equivalent to
                                    # the first five percent of noise duration
    windowStart = signal.hann( 2 * fivePercentSample )
    fullWindow = np.concatenate( ( windowStart[0:fivePercentSample], \
                                  np.ones( int( noiseSamples \
                                               - fivePercentSample ) ) ) )
    newNoise = fullWindow * inputNoise
    return newNoise	



def impulse(samplingRate = None,
			fftDegree = None):
    """
    Generates a normalized impulse signal at time zero,
    with zeros to fill the time length
    """
    if samplingRate is None: samplingRate = default.samplingRate
    if fftDegree is None: fftDegree = default.fftDegree
    
    numSamples = 2**fftDegree
    impulseSignal = (numSamples / samplingRate) \
                    * np.ones(numSamples) + 1j * np.random.randn(numSamples)
    impulseSignal = np.real( np.fft.ifft( impulseSignal ) )
    impulseSignal = impulseSignal / max( impulseSignal )
    newImpulse = SignalObj(signalArray=impulseSignal,domain='time',samplingRate=samplingRate )
    return newImpulse


	
def measurement(kind = 'playrec',
                samplingRate = None,
                freqMin = None,
                freqMax = None,
                device = None,
                inChannel = None,
                outChannel = None,
                *args,
                **kwargs,
                ):
    """
	Generates a measurement object of type Recording, Playback and Recording,
	Transferfunction, with the proper initiation arguments, a sampling rate,
	frequency limits, audio input and output devices and channels
	
		>>> pytta.generate.measurement(kind,
                                       [lengthDomain,
                                       fftDegree,
                                       timeLength,
                                       excitation],
                                       samplingRate,
                                       freqMin,
                                       freqMax,
                                       device,
                                       inChannel,
                                       outChannel,
                                       comment
                                       )
	
    The parameters between brackets are different for each value of the (kind)
    parameter.
    
	>>> msRec = pytta.generate.measurement(kind='rec')
	>>> msPlayRec = pytta.generate.measurement(kind='playrec')
	>>> msFRF = pytta.generate.measurement(kind='frf')
	
	The input arguments may be different for each measurement kind.
	
	Options for (kind='rec'):
			
			- lengthDomain: 'time' or 'samples', defines if the recording length will be set by time length, or number of samples
			- timeLength: [s] used only if (domain='time'), set the duration of the recording, in seconds;
			- fftDegree: represents a power of two value that defines the number of samples to be recorded:
							
								>>> numSamples = 2**fftDegree
							
			- samplingRate: [Hz] sampling frequency of the recording;
			- freqMin: [Hz] smallest frequency of interest;
			- freqMax: [Hz] highest frequency of interest;
			- device: audio I/O device to use for recording;
			- inChannel: list of active channels to record;
			- comment: any commentary about the recording.


    Options for (kind='playrec'):
			
			- excitation: object of SignalObj class, used for the playback. 
			- samplingRate: [Hz] sampling frequency of the recording;
			- freqMin: [Hz] smallest frequency of interest;
			- freqMax: [Hz] highest frequency of interest;
			- device: audio I/O device to use for recording;
			- inChannel: list of active channels to record;
			- outChannel: list of active channels to send the playback signal, for M channels it is mandatory for the excitation signal to have M columns in the timeSignal parameter.
			- comment: any commentary about the recording.

	Options for (kind='frf'):

			Same as for (kind='playrec')
    """
##%% Default Parameters
    if freqMin is None: freqMin = default.freqMin
    if freqMax is None: freqMax = default.freqMax
    if samplingRate is None: samplingRate = default.samplingRate
    if device is None: device = default.device
    if inChannel is None: inChannel = default.inChannel
    if outChannel is None: outChannel = default.outChannel

##%% Kind REC
    if kind in ['rec','record','recording','r']:
        recordObj = RecMeasure(samplingRate = samplingRate,
                            freqMin = freqMin,
                            freqMax = freqMax,
                            device = device,
                            inChannel = inChannel,
                            **kwargs,
                            )
        if ('lengthDomain' in kwargs) or args:
            if kwargs.get('lengthDomain') == 'time':
                recordObj.lengthDomain = 'time'
                try:
                    recordObj.timeLength = kwargs.get('timeLength')
                except:
                    recordObj.timeLength = default.timeLength
            elif kwargs.get('lengthDomain') == 'samples':
                recordObj.lengthDomain = 'samples'
                try:
                    recordObj.fftDegree = kwargs.get('fftDegree')
                except:
                    recordObj.fftDegree = default.fftDegree
        else:
            recordObj.domain = 'samples'
            recordObj.fftDegree = default.fftDegree
        return recordObj
	
##%% Kind PLAYREC    
    elif kind in ['playrec','playbackrecord','pr']:
        if ('excitation' in kwargs) or args:
            signalIn = kwargs.get('excitation') or args[0]
            kwargs.pop('excitation', None)
        else:
            signalIn = sweep(samplingRate = samplingRate,
                             freqMin = freqMin,
                             freqMax = freqMax,
                             **kwargs)
			
        playRecObj = PlayRecMeasure(excitation = signalIn,
                                    device = device,
                                    inChannel = inChannel,
                                    outChannel = outChannel,
                                    **kwargs)
        return playRecObj
	
##%% Kind FRF    
    elif kind in ['tf','frf','transferfunction','freqresponse']:
        if ('excitation' in kwargs) or args:
            signalIn = kwargs.get('excitation') or args[0]
            kwargs.pop('excitation', None)
        else:
            signalIn = sweep(samplingRate = samplingRate,
                             freqMin = freqMin,
                             freqMax = freqMax,
                             **kwargs)

        frfObj = FRFMeasure(excitation = signalIn,
                            device = device,
                            inChannel = inChannel,
                            outChannel = outChannel,
                            **kwargs
                            )
        return frfObj