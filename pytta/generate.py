# -*- coding: utf-8 -*-
"""
Generate
=========
  
@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
- Matheus Lazarin Alberto, mtslazarin@gmail.com

@Última modificação: 27/10/18

"""

#%%
from pytta.classes import signalObj, RecMeasure, FRFMeasure, PlayRecMeasure
from pytta.properties import default
from scipy import signal
import numpy as np

def sweep(Finf = default['freqMin'],
          Fsup = default['freqMax'],
          Fs = default['samplingRate'],
          fftDeg = default['fftDegree'],
          startmargin = default['startMargin'],
          stopmargin = default['stopMargin'],
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
    Flim = np.array([Finf/(2**(1/6)), min(Fsup*(2**(1/6)),Fs/2)]); # frequency limits [Hz]
    Ts = 1/Fs # [s] sampling period
    Nstop = stopmargin*Fs # [samples] initial silence number of samples
    Nstart = startmargin*Fs # [samples] ending silence number of samples
    Nmargin = Nstart + Nstop # [samples] total silence number of samples
    N = 2**fftDeg # [samples] full signal number of samples
    Nsweep = N - Nmargin +1 # [samples] actual sweep number of samples
    Tsweep = Nsweep/Fs # [s] sweep's time length
    tsweep = np.arange(0,Tsweep,Ts) # [s] sweep time vector
    if tsweep.size > Nsweep: tsweep = tsweep[0:int(Nsweep)] # adjust length
    sweept = 0.8*signal.chirp(tsweep, Flim[0], Tsweep, Flim[1],\
                              'logarithmic', phi=-90) # sweep, time domain
    sweept = __do_sweep_windowing(sweept,tsweep,Flim,Finf,Fsup,windowing) # fade in and fade out
    xt = np.concatenate( (np.zeros( int(Nstart) ),\
                          sweept,\
                          np.zeros( int(Nstop) ) ) ) # add initial and ending sileces
    if xt.size != N: xt = xt[0:int(N)] # adjust length
    x = signalObj(xt,'time',Fs) # transforms into a pytta signalObj
    x.Flim = Flim # pass on the frequency limits considering the fade in and fade out
    return x

def __do_sweep_windowing(in_signal,
                        tsweep,
                        Flim, Finf, Fsup,
                        win):
    """
    Applies a fade in and fade out that are minimum at the chirp start and end,
    and maximum between the time intervals corresponding to Finf and Fsup.
    
    """
    
    fsweep = Flim[0]*((Flim[1]/Flim[0])**(1/max(tsweep)))**tsweep # frequencias em função do tempo: freq(t)
    a1 = np.where(fsweep<=Finf)
    a2 = np.where(fsweep<=Fsup)
    a1 = a1[-1][-1]
    a2 = len(fsweep) - a2[-1][-1]
    wins = signal.hann(2*a1)
    winf = signal.hann(2*a2)
    win = np.concatenate((wins[0:a1], np.ones(int(len(fsweep)-a1-a2+1)), winf[a2:-1]))
    new_signal = win*in_signal
    return new_signal
 
 
 
def noise(kind = 'white',
			 Fs = default['samplingRate'],
          fftDeg = default['fftDegree'],
          startmargin = default['startMargin'],
          stopmargin = default['stopMargin'],
          windowing = 'hann'):
	"""
	Generates a noise of kind White, Pink or Blue, with a silence at the
	begining and ending of the signal, plus a fade in to avoid abrupt speaker
	excursioning. All noises have normalized amplitude.
	
		White noise is generated with a numpy.randn;
	
		Pink noise is still in progress;
	
		Blue noise is still in progress;
	
		Flat noise is generated as a frequency constant magnitude plus a 
		numpy.randn complex phase, then ifft;
	
	"""
	
	Nstart = int(startmargin*Fs) # [samples] Starting silence number of samples
	Nstop = int(stopmargin*Fs) # [samples] Ending silence number of samples
	Nmargin = Nstart + Nstop # [samples] total silence number of samples
	N = 2**fftDeg # [samples] full signal number of samples
	Nnoise = int(N - Nmargin) # [samples] Actual noise number of samples
	if kind.upper() in ['WHITE','FLAT']:
		noiseSignal = np.random.randn(Nnoise)
#	elif kind.upper() == 'PINK':
#		noiseSignal = np.randn(Nnoise)
#		noiseSignal = noiseSignal/max(abs(noiseSignal))
#		noiseSignal = _do_pink_filtering(noiseSignal)
#	elif kind.upper() == 'BLUE':
#		noiseSignal = np.randn(Nnoise)
#		noiseSignal = noiseSignal/max(abs(noiseSignal))
#		noiseSignal = _do_blue_filtering(noiseSignal)

	noiseSignal = _do_noise_windowing(noiseSignal,Nnoise,windowing)
	noiseSignal = noiseSignal/max(abs(noiseSignal))
	signal = np.concatenate((np.zeros(Nstart),noiseSignal,np.zeros(Nstop)))
	signal = signalObj(signal,'time',Fs)
	return signal

def _do_noise_windowing(in_signal,
                        Nnoise,
                        win):
	a = int((5/100)*(Nnoise))
	wins = signal.hann(2*a)
	win = np.concatenate((wins[0:a], np.ones(int(Nnoise-a))))
	new_signal = win*in_signal
	return new_signal	


def impulse(Fs = default['samplingRate'],
				fftDeg = default['fftDegree']):
	N = 2**fftDeg
	impulseSignal = (N/Fs)*np.ones(N) + 1j*np.random.randn(N)
	impulseSignal = np.real(np.fft.ifft(impulseSignal))
	impulseSignal = impulseSignal/max(impulseSignal)
	signal = signalObj(impulseSignal,'time',Fs)
	return signal
	
def measurement(kind = 'playrec',*args,
                Fs = default['samplingRate'],
                Finf = default['freqMin'],
                Fsup = default['freqMax'],
                device = default['device'],
                inch = default['inch'],
                outch = default['outch'],**kwargs):
    
	"""
	Generates a measurement object of type Recording, Playback and Recording,
	Transferfunction, with the proper initiation arguments, a sampling rate,
	frequency limits, audio input and output devices and channels
	
	>>> msRec = pytta.generate.measurement('rec')
	>>> msPlayRec = pytta.generate.measurement('playrec')
	>>> msFRF = pytta.generate.measurement('frf')
	
	The input arguments may be different for each measurement kind.
	
	"""
	
	if kind in ['rec','record','recording','r']:
		recObj = RecMeasure(Fs=Fs,
                                  Finf=Finf,
                                  Fsup=Fsup,
                                  device=device,
                                  inch=inch,**kwargs)
		if ('domain' in kwargs) or args:
			if kwargs.get('domain') == 'time' or args[0]=='time':
				recObj.domain = 'time'
				recObj.timeLen = kwargs.get('timeLen') or args[1]
			elif kwargs.get('domain') == 'samples' or args[0]=='samples':
				recObj.domain = 'samples'
				recObj.fftDeg = kwargs.get('fftDeg') or args[1]
		else:
			recObj.domain = 'samples'
			recObj.fftDeg = default['fftDegree']
		return recObj
	
	elif kind in ['playrec','playbackrecord','pr']:
		if ('excitation' in kwargs) or args:
			signalIn=kwargs.get('excitation') or args[0]
			kwargs.pop('excitation',None)
		else:
			signalIn = sweep(Fs=Fs,Finf=Finf,Fsup=Fsup,**kwargs)
			
		prObj = PlayRecMeasure(signalIn,
                                     device=device,
                                     inch=inch,
                                     outch=outch,
                                     **kwargs)
		return prObj
	
	elif kind in ['tf','frf','transferfunction','freqresponse']:
		if ('excitation' in kwargs) or args:
			signalIn=kwargs.get('excitation') or args[0]
			kwargs.pop('excitation',None)
		else:
			signalIn = sweep(Fs=Fs,Finf=Finf,Fsup=Fsup,**kwargs)
			
		frfObj = FRFMeasure(excitation=signalIn,
                                device=device,
                                inch=inch,
                                outch=outch,
                                **kwargs)
		return frfObj

