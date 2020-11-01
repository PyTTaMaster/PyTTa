.. PyTTa documentation master file, created by
   sphinx-quickstart on Sat May 23 13:14:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. currentmodule:: pytta

API documentation
=================

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
        
    * pytta.rooms:
        room acoustics parameters calculation according to ISO 3382-1;
        
    * pytta.iso3741:
        calculations according to the standard;

.. toctree::
   classes/index
   utils/index
   apps/index
   functions
   generate
   rooms
   properties
