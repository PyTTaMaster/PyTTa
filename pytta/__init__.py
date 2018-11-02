#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                     PyTTa
    Object Oriented Python in Technical Acoustics

                    __init__
  
@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
- Matheus Lazarin Alberto, mtslazarin@gmail.com

@Última modificação: 29/10/18


PyTTa

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
    mandatory thing, but you should.

"""

#%% Importing .py files as submodules
from . import generate
from . import properties
from .classes import signalObj, RecMeasure, PlayRecMeasure, FRFMeasure

__version__ = '0.0.0b' # package version

# package submodules and scripts to be called as pytta.something
__all__ = ['generate','properties','RecMeasure',\
           'PlayRecMeasure','FRFMeasure','signalObj'] 