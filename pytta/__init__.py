#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Oriented Python in Technical Acoustics
==============================================

Autores:
	João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
	Matheus Lazarin Alberto, mtslazarin@gmail.com


PyTTa:

    This is a package developed to perform acoustic and vibration measurements
    and signal analysis. In order to provide such functionalities we require a
    few packages to be installed:
    
    - Numpy
    - Scipy
    - Matplotlib
    - PortAudio API
    - Sounddevice
    - PyFilterbank (future release)
    
    We also recommend using the Anaconda Python distribution, it's not a
    mandatory issue, but you should.
	 
	To begin, try:
		 
		 >>> import pytta
		 >>> pytta.properties.default
		 >>> pytta.list_devices()

PyTTa user intended classes:
    
    >>> pytta.signalObj()
    >>> pytta.generate.sweep()
    >>> pytta.generate.noise()
    >>> pytta.generate.impulse()
    >>> pytta.generate.measurement()

For further information see the specific method documentation
"""

#%% Importing .py files as submodules
from .functions import read_wav, write_wav, merge, list_devices, fftconvolve, super_convolve, resample
from . import generate
from . import properties
from .classes import signalObj, RecMeasure, PlayRecMeasure, FRFMeasure

__version__ = '0.0.0a2' # package version

# package submodules and scripts to be called as pytta.something
__all__ = ['generate','properties','merge','fftconvolve','read_wav','write_wav','list_devices',\
           'super_convolve','resample','RecMeasure','PlayRecMeasure','FRFMeasure','signalObj'] 
