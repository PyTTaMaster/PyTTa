#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Applications sub-package.

This sub-package contains applications built from the toolbox's
functionalities. The apps are called from the top-level (e.g. pytta.roomir).

Available apps:
---------------
    
    * pytta.roomir:
        Room impulsive response acquisition and post-processing;

Created on Sun Jun 23 15:02:03 2019

@author: Matheus Lazarin - matheus.lazarin@eac.ufsm.br

"""

from pytta.apps import roomir

__all__ = ['roomir']
