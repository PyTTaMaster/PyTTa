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
		 >>> pytta.Default()
		 >>> pytta.list_devices()

    You can find out everything available reading the submodules documentation:
        
        >>> pytta.classes
        >>> pytta.generate
        >>> pytta.functions
        >>> pytta.properties

For further information, check the specific module, class, method or function documentation.    
"""

#%% Importing .py files as submodules
from . import properties

# Instantiate the Default parameters to be loaded by other methods and function calls
Default = properties.Default()

from .classes import signalObj, RecMeasure, PlayRecMeasure, FRFMeasure
from .functions import read_wav, write_wav, merge, list_devices, fftconvolve, finddelay, corrcoef, resample
from . import generate

#Default = properties.Default

__version__ = '0.0.0a2' # package version

# package submodules and scripts to be called as pytta.something
__all__ = [# Submodules
           'generate',
           
           # Functions
           'merge',
           'fftconvolve',
           'read_wav',
           'write_wav',
           'list_devices',
           'finddelay',
           'resample',
           'corrcoef',
           
           # Classes
           'RecMeasure',
           'PlayRecMeasure',
           'FRFMeasure',
           'signalObj',
           
           # Objects
           'Default',
           ] 
