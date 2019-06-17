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
        >>> pytta.default()
        >>> pytta.list_devices()

    You can find out everything available reading the submodules documentation:

        >>> pytta.classes
        >>> pytta.generate
        >>> pytta.functions
        >>> pytta.properties

For further information, check the specific module, class, method or function
documentation.
"""

# Importing .py files as submodules
from . import properties

# Instantiate the Default parameters to be loaded by other
# methods and function calls
default = properties.Default()
units = properties.units

from .classes import SignalObj, ImpulsiveResponse,\
                    RecMeasure, PlayRecMeasure, FRFMeasure, Streaming
from .functions import read_wav, write_wav, merge, list_devices,\
                    fft_convolve, find_delay, corr_coef, resample, peak_time,\
                    save, load
from .filter import OctFilter, fractional_octave_frequencies
from . import generate

__version__ = '0.1.0rc'  # package version

# package submodules and scripts to be called as pytta.something
__all__ = [  # Submodules
           'generate',

           # Functions
           'merge',
           'fft_convolve',
           'read_wav',
           'write_wav',
           'list_devices',
           'find_delay',
           'resample',
           'corr_coef',
           'peak_time',
           'save',
           'load',
           'fractional_octave_frequencies',

           # Classes
           'RecMeasure',
           'ImpulsiveResponse',
           'PlayRecMeasure',
           'FRFMeasure',
           'SignalObj',
           'OctFilter',

           # Objects
           'default']
