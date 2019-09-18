# -*- coding: utf-8 -*-
"""
Object Oriented Python in Technical Acoustics
==============================================

Autores:
    João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
    Matheus Lazarin Alberto, matheus.lazarin@eac.ufsm.br


PyTTa:

    This is a package developed to perform acoustic and vibration measurements
    and signal analysis. In order to provide such functionalities we require a
    few packages to be installed:

    - Numpy
    - Scipy
    - Matplotlib
    - PortAudio API
    - Sounddevice

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
import pytta.properties as properties

# Instantiate the Default parameters to be loaded by other
# methods and function calls
default = properties.Default()

from pytta.classes import SignalObj, ImpulsiveResponse,\
                          RecMeasure, PlayRecMeasure, FRFMeasure,\
                          Streaming,\
                          OctFilter, weighting
from pytta.functions import read_wav, write_wav, merge, list_devices,\
                    fft_convolve, find_delay, corr_coef, resample, peak_time,\
                    save, load, fft_degree

import pytta.rooms as rooms
import pytta.apps.roomir as roomir
import pytta.generate as generate
import pytta.h5utilities

__version__ = '0.1.0b4'  # package version

# package submodules and scripts to be called as pytta.something
__all__ = [  # Apps
           'roomir',

           # Submodules
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
           'h5save',
           'load',
           'h5load',
           'weighting',
           'fft_degree',

           # Classes
           'RecMeasure',
           'ImpulsiveResponse',
           'PlayRecMeasure',
           'FRFMeasure',
           'SignalObj',
           'OctFilter',
           'Streaming',

           # Objects
           'default']
