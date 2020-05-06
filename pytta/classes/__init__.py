"""
Classes:
---------

    This submodule is mainly the means to an end. PyTTa is made intended to be
    user friendly, the manipulation of the classes are documented here, but
    their instantiation should be used through the <generate> submodule:

        >>> pytta.generate.sweep()
        >>> pytta.generate.noise()
        >>> pytta.generate.measurement('playrec')
        >>> pytta.generate.measurement('rec', lengthDomain='time', timeLen=5)

    This way, the default settings will be loaded into any object instantiated.

    User intended classes:

        >>> pytta.SignalObj()
        >>> pytta.RecMeasure()
        >>> pytta.PlayRecMeasure()
        >>> pytta.FRFMeasure()

    For further information see the specific class, or method, documentation

"""


from .signal import SignalObj, ImpulsiveResponse
#from .signal import plot_SignalObjs
from .measurement import RecMeasure, PlayRecMeasure, FRFMeasure
from .streaming import Streaming, Recorder, Monitor
from .filter import OctFilter, weighting
from .analysis import Analysis

__all__ = [ # Classes
           'SignalObj',
           'ImpulsiveResponse',
           'RecMeasure',
           'PlayRecMeasure',
           'FRFMeasure',
           'Streaming',
           'Monitor',
           'Recorder',
           'OctFilter',
           'weighting',
           'Analysis']#,

        #    # Functions
        #   'plot_SignalObjs']
