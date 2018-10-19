#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                     PyTTa
    Object Oriented Python in Technical Acoustics

                    __init__
  
@Autores:
- Matheus Lazarin Alberto, mtslazarin@gmail.com

@Última modificação: 16/10/18

EM EDIÇÃO!!! TENTANDO REMOVER CRIAÇÃO DO SWEEP DO FRFMEASURE
E CRIAR UMA NOVA FUNÇÃO PARA PROJETO DE SINAIS
"""
#%% importing bibs
import numpy as np
import matplotlib.pyplot as plot
import scipy.signal as signal
import scipy.io as io
import sounddevice as sd
#%% signalobj
class signalobj:
    def __init__(self,t=np.array([0]),Fs=44100):
        self.Fs = Fs # [Hz] sample rate
        self._t = t # [-] signal in time domain
        self.N = np.size(self.t) # [-] number of samples
        self.Tp = self.N/self.Fs # [s] signal time lenght
        self.time = np.arange(0, self.Tp, 1/self.Fs) # [s] time vector
        self.freq = np.linspace(0, (self.N-1)*self.Fs/self.N, self.N)# [Hz] frequency vector
        self._jw = 2/self.N*np.fft.fft(self.t) # [-] signal in frequency domain
        self.Flim = np.array([20, 20000])
        self.comment = ['No comments.']
    
    def __truediv__(self, other):
        # Permit dividir dois signalobj e obter a divisão dos seus espectros em um novo signalobj
        resul = signalobj(Fs=self.Fs)
        resul.jw = self.jw/other.jw
        return resul
        
    @property # when t is called as object.t it returns the nparray 
    def t(self):
        return self._t
    @t.setter
    def t(self,newt): # when t is called as object.t(newt) it returns the nparray
        self._t = newt
        self.N = np.size(self.t) # [-] number of samples
        self.Tp = self.N/self.Fs # [s] signal time lenght
        self.time = np.arange(0, self.Tp, 1/self.Fs) # [s] time vector
        self.freq = np.linspace(0, (self.N-1)*self.Fs/self.N, self.N)# [Hz] frequency vector
        self._jw = 2/self.N*np.fft.fft(self.t) # [-] signal in frequency domain
        
    @property
    def jw(self):
        return self._jw
    @jw.setter
    def jw(self,newjw):
        self._jw = newjw
        self._t = np.fft.ifft(self.jw)
        self.N = np.size(self.t) # [-] number of samples
        self.Tp = self.N/self.Fs # [s] signal time lenght
        self.time = np.arange(0, self.Tp, 1/self.Fs) # [s] time vector
        self.freq = np.linspace(0, (self.N-1)*self.Fs/self.N, self.N)# [Hz] frequency vector
        
    def play(self):
        sd.play(self.t,self.Fs)
        
#   def plot(self): # TO DO
#        ...

#    def plot_time(self): # TO DO
#       ...
        
    def plot_freq(self):
        plot.figure(figsize=(10,5))
        plot.semilogx(self.freq,20*np.log10(np.abs(self.jw)),label='H(jw)')
        Hjw_suave = signal.savgol_filter(abs(self.jw),31,3);
        plot.semilogx(self.freq,20*np.log10(np.abs(Hjw_suave)),label='H(jw) suavizada')
        plot.axis([self.Flim[0], self.Flim[1], 1.05*np.min(20*np.log10(np.abs(self.jw))), 3])
        plot.xlabel(r'$Frequência$ [Hz]')
        plot.ylabel(r'$|H(jw)|$ dB - ref.: 1 [-]')
        plot.legend()
#%% generate
def generate(Fs,Finf,Fsup,fftdeg,stopmargin):
#    self.Fs = Fs # [Hz] sampling frequency
#    self.Finf = Finf # [Hz] frequency limits 
#    self.Fsup = Fsup # [Hz] frequency limits
#    self.Flim = np.array([self.Finf, self.Fsup]); # frequency limits [Hz]
#    self.fftdeg = fftdeg # 2^(fftdeg) [samples]
#    self.stopmargin = stopmargin # [s] avarage system reponse time
#    self.Ts = 1/self.Fs # [s] sampling period
#    self.Nstopmargin = self.stopmargin*self.Fs # [samples] stpomargin's number of samples
#    self.N = 2**self.fftdeg + self.Nstopmargin # [samples] measurement number of samples
#    self.Nsweep = 2**self.fftdeg # [samples] sweep's number of samples
#    self.Tsweep = self.Nsweep/self.Fs # [s] sweep's time length
#    self.tsweep = np.arange(0,self.Tsweep,self.Ts) # [s] sweep time vector
#    if self.tsweep.size > self.Nsweep: self.tsweep = self.tsweep[0:int(self.Nsweep)]
#    self.sweept = 0.8*signal.chirp(self.tsweep, self.Flim[0], self.Tsweep, self.Flim[1], 'logarithmic') # x(t) - in signal: sweep
#    self.x = np.concatenate((self.sweept,np.zeros(int(self.Nstopmargin))))
#    self.x = signalobj(self.x,self.Fs)
#    self.x.Flim = self.Flim
#    return self.x
    Flim = np.array([Finf, Fsup]); # frequency limits [Hz]
    Ts = 1/Fs # [s] sampling period
    Nstopmargin = stopmargin*Fs # [samples] stpomargin's number of samples
    N = 2**fftdeg + Nstopmargin # [samples] measurement number of samples
    Nsweep = 2**fftdeg # [samples] sweep's number of samples
    Tsweep = Nsweep/Fs # [s] sweep's time length
    tsweep = np.arange(0,Tsweep,Ts) # [s] sweep time vector
    if tsweep.size > Nsweep: tsweep = tsweep[0:int(Nsweep)]
    sweept = 0.8*signal.chirp(tsweep, Flim[0], Tsweep, Flim[1], 'logarithmic') # x(t) - in signal: sweep
    xt = np.concatenate((sweept,np.zeros(int(Nstopmargin))))
    x = signalobj(xt,Fs)
    x.Flim = Flim
    return x

#%% frfmeasure
class frfmeasure: #criar opção de entrada de um signalobj p/ sinal de excitação
    #implementar check de entradas
    def __init__(self,device,inch,outch,Fs,Finf,Fsup,comment,x):        #    def __init__(self,device=2,inch=[4],outch=[7],Fs=44100,Finf=20,Fsup=20000,fftdeg=16,stopmargin=0.5,comment=['No comments.']):        
        # measurement preferences
        self.device = device # device number. For device list use sounddevice.query_devices()
        # device = [0,1]; # [in,out] device number
        self.Fs = Fs
        self.inch = inch # input channels
        self.outch = outch # output channels
        self.Finf = Finf
        self.Fsup = Fsup
        self.comment = comment # comment about the measurement
        self.x = x # excitation signalobj
        self.N = self.x.N
        #%% measurement setup
        self.Flim = np.array([self.Finf, self.Fsup]); # frequency limits [Hz]
        self.Ts = 1/self.Fs # [s] sampling period
        self.Tp = self.x.N/self.Fs # [s] measurement time
        self.t = np.arange(0, self.Tp, 1/self.Fs) # [s] time vector
        self.freq = np.linspace(0,(self.N-1)*self.Fs/self.N, self.N)# [Hz] frequency vector
        # (N-1)*(Fs/N) = Fs - Fs/N; Fs/N is the spectrum resolution (delta F)
    #%%
    def run(self):
        self.yt = sd.playrec(self.x.t, self.Fs, input_mapping=self.inch,output_mapping=self.outch,device=self.device) # y_all(t) - out signal: x(t) conv h(t)
        sd.wait()   
        self.yt = np.squeeze(self.yt) # turn column array into line array
        self.y = signalobj(self.yt,self.Fs)
        print('x(t) (playback) max level: ', 20*np.log10(max(self.x.t)), 'dBFs - ref.: 1 [-]')
        print('y(t) (input) max level: ', 20*np.log10(max(self.y.t)), 'dBFs - ref.: 1 [-]')
        self.h = self.y/self.x
        self.h.Flim = self.Flim
        return self.h