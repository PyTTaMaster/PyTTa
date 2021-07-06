# -*- coding: utf-8 -*-
"""
This is a package developed to perform acoustic and vibration measurements
and signal analysis.

To get started, try:

    >>> import pytta

    >>> pytta.default()
    >>> pytta.list_devices()

    >>> mySignal = pytta.generate.sweep()
    >>> mySignal.plot_freq()  # same as pytta.plot_freq(mySignal)

Our current dependencies are:

    * Numpy
    * Scipy
    * Matplotlib
    * Numba
    * h5py
    * SoundDevice
    * SoundFile

PyTTa is now a multi-paradigm toolbox, which may change in the future. We aim
from now on to be more Pythonic, and therefore more object-oriented. For now,
we have classes and functions operating together to provide a working
environment for acoustical measurements and post-processing.

The toolbox' structure is presented next.


Sub-packages:
-------------

    * pytta.classes:
        main classes intended to do measurements (Rec, PlayRec and
        FRFMeasurement), handle signals/processed data (SignalObj and
        Analysis), handle streaming functionalities (Monitor and Streaming),
        filter (OctFilter), communicate with some hardware (LJU3EI1050),
        and also intended for new features whose should have an
        object-oriented implementation. The main classes are called from the
        toolbox's top-level (e.g. pytta.SignalObj, pytta.Analysis, ...);

    * pytta.utils:
        contains simple tools which help to keep things modularized and make
        more accessible the reuse of some operations. Intended to hold tools
        (classes and functions) whose operate built-in python classes, NumPy
        arrays, and other stuff not contained by the pytta.classes subpackage;

    * pytta.apps:
        applications built from the toolbox's functionalities. The apps
        are called from the top-level (e.g. pytta.roomir);


Modules:
--------

    * pytta.functions:
        assistant functions intended to manipulate and visualize multiple
        signals and analyses stored as SignalObj and Analysis objects. These
        functions are called from the toolbox's top-level (e.g.
        pytta.plot_time, pytta.plot_waterfall, ...);

    * pytta.generate:
        functions for signal synthesis and measurement configuration;

    * pytta.rooms (DEPRECATED):
        room acoustics parameters calculation according to ISO 3382-1;

    * pytta.iso3741:
        calculations according to the standard;


You can find usage examples in the toolbox's examples folder.

For further information, check the sub-packages' and modules' documentation.

The documentation is also available at:

    https://pytta.readthedocs.io/


Created on Fri May 25 2018

@authors:
    João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
    Matheus Lazarin, matheus.lazarin@eac.ufsm.br

"""

# Importing .py files as submodules
from ._properties import default

# Instantiate the Default parameters to be loaded by other
# methods and function calls

from .classes import SignalObj, ImpulsiveResponse,\
    Measurement, RecMeasure, PlayRecMeasure, FRFMeasure,\
    Streaming, Monitor,\
    OctFilter, weighting,\
    Analysis, RoomAnalysis

from . import _h5utils
from . import _plot
from . import generate
from . import utils
from . import rooms
from . import iso3741

from .functions import read_wav, write_wav, merge, split, \
    list_devices, print_devices, get_device_from_user,\
    fft_convolve, find_delay, corr_coef, resample,\
    peak_time, save, load, plot_time, plot_time_dB, plot_freq,\
    plot_bars, plot_spectrogram, plot_waterfall, SPL

# Must go on v0.1.0 (DEPRECATED)
from .functions import fft_degree

from .apps import roomir

__version__ = '0.1.1'  # package version

# package submodules and scripts to be called as pytta.something
__all__ = [  # Apps
           'roomir',

           # Submodules
           'generate',
           'utils',
           'iso3741',
           'rooms',

           # Functions
           'merge',
           'split',
           'fft_convolve',
           'read_wav',
           'write_wav',
           'SPL',
           'list_devices',
           'print_devices',
           'get_device_from_user',
           'find_delay',
           'resample',
           'corr_coef',
           'peak_time',
           'save',
           'load',
           'weighting',
           'plot_time',
           'plot_time_dB',
           'plot_freq',
           'plot_bars',
           'plot_spectrogram',
           'plot_waterfall',

           # Classes
           'RecMeasure',
           'PlayRecMeasure',
           'FRFMeasure',
           'SignalObj',
           'ImpulsiveResponse',
           'Analysis',
           'RoomAnalysis',
           'OctFilter',
           'Monitor',
           'Streaming',

           # Objects
           'default']
