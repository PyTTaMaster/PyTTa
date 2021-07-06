"""
Classes/Core sub-package.

Main classes intended to do measurements (Rec, PlayRec and FRFMeasurement),
handle signals/processed data (SignalObj and Analysis), handle streaming
functionalities (Monitor and Streaming), filter (OctFilter), communicate with
some hardware (LJU3EI1050), and also intended for new features whose should
have an object-oriented implementation. Called from the toolbox's top-level
(e.g. pytta.SignalObj, pytta.Analysis, ...).

Available classes:

    * SignalObj
    * ImpulsiveResponse
    * Analysis
    * RecMeasure
    * PlayRecMeasure
    * FRFMeasure
    * Streaming
    * Monitor
    * OctFilter

The instantiation of some classes should be done through the 'generate'
submodule, as for measurements and signal synthesis. This way, the default
settings will be loaded into those objects. E.g.:

    >>> mySweepSignalObj = pytta.generate.sweep()
    >>> myNoiseSignalObj = pytta.generate.random_noise()
    >>> myMeasurementdObj1 = pytta.generate.measurement('playrec')
    >>> myMeasurementdObj2 = pytta.generate.measurement('rec',
    >>>                                                 lengthDomain='time',
    >>>                                                 timeLen=5)

For further information, see the specific class documentation.

"""


from .signal import SignalObj, ImpulsiveResponse
#from .signal import plot_SignalObjs
from .measurement import Measurement, RecMeasure, PlayRecMeasure, FRFMeasure
from .streaming import Streaming, Monitor
from .filter import OctFilter, weighting
from .analysis import Analysis, RoomAnalysis

__all__ = [# Classes
           'SignalObj',
           'ImpulsiveResponse',
           'Analysis',
           'RoomAnalysis',
           'RecMeasure',
           'PlayRecMeasure',
           'FRFMeasure',
           'Streaming',
           'Monitor',
           'OctFilter',
           # Functions
           'weighting']
