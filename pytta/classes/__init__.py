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


from pytta.classes.signal import SignalObj, ImpulsiveResponse
from pytta.classes.measurement import RecMeasure, PlayRecMeasure, FRFMeasure
from pytta.classes.streaming import Streaming
from pytta.classes.filter import OctFilter, weighting
from pytta.classes.analysis import Analysis

__all__ = ['SignalObj', 'ImpulsiveResponse',
           'RecMeasure', 'PlayRecMeasure', 'FRFMeasure',
           'Streaming',
           'OctFilter', 'weighting',
           'Analysis']
