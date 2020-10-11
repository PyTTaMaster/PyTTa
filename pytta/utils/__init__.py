#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities sub-package.

Contains simple tools which help to keep things modularized and make
accessible the reuse of some operations. Intended to hold tools (classes and
functions) whose operate built-in python classes, NumPy arrays, and other
stuff not contained by the pytta.classes sub-package.


Available modules:
------------------

    * colore
    * freq
    * maths
    

Created on Tue May 5 00:31:42 2020

@author: João Vitor G. Paes

"""

from .colore import ColorStr, colorir, pinta_texto, pinta_fundo
from .freq import *
from .maths import *

