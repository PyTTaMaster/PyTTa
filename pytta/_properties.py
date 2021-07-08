# -*- coding: utf-8 -*-
"""
As to provide an user friendly signal measurement package, a few default
values where assigned to the main classes and functions.

These values where set using a dict called "default", and are passed to all
PyTTa functions through the Default class object

    >>> import pytta
    >>> pytta.default()

The default values can be set differently using both declaring method, or
the set_default() function

    >>> pytta.default.propertyName = propertyValue
    >>> pytta.default.set_defaults(propertyName1 = propertyValue1,
    >>>                            ... ,
    >>>                            propertyNameN = propertyValueN
    >>>                            )

The main difference is that using the set_default() function, a list of
properties can be set at the same time

The default device start as the one set default at the user's OS. We
recommend changing it's value to the desired audio in/out device, as it
can be identified using list_devices() method

    >>> pytta.list_devices()

@author:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

"""
import sounddevice as sd
from typing import Any


__default_device = [sd.default.device[0], sd.default.device[1]]
""" Used only to hold the default audio I/O device at pytta import time"""


default_ = {'samplingRate': 44100,
            'lengthDomain': 'samples',
            'fftDegree': 18,
            'timeLength': 10,
            'integration': 0.125,
            'freqMin': 20,
            'freqMax': 20000,
            'device': __default_device,
            'inChannel': [1],
            'outChannel': [1],
            'stopMargin': 0.7,
            'startMargin': 0.3,
            'comment': 'No comments.',
            }


class Default(object):
    """Holds default parameter values."""

    _samplingRate = []
    _lengthDomain = []
    _fftDegree = []
    _timeLength = []
    _integration = []
    _freqMin = []
    _freqMax = []
    _device = []
    _inChannel = []
    _outChannel = []
    _stopMargin = []
    _startMargin = []
    _comment = []
    _instance = None

    def __init__(self):
        """Singleton with properties used across PyTTa."""
        for name, value in default_.items():
            vars(self)['_'+name] = value
        return

    def __new__(cls):
        """Instance created only once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __setattr__(self, name: str, value: Any):
        """
        Allow assign operation on attributes.

        Parameters
        ----------
        name : str
            Attribute name as string.
        value : Any
            New attribute value.

        Raises
        ------
        AttributeError
            If attribute does not exist.

        Returns
        -------
        None.

        """
        if name in dir(self) and name != 'device':
            vars(self)['_'+name] = value
        elif name in ['device', 'devices']:
            self.set_defaults(device=value)
        else:
            raise AttributeError('There is no default settings for ' + repr(name))

    def __call__(self):
        """Good view of attributes."""
        for name, value in vars(self).items():
            if len(name) <= 8:
                print(name[1:]+'\t\t =', value)
            else:
                print(name[1:]+'\t =', value)

    def set_defaults(self, **namevalues):
        """
        Change the values of the "Default" object's properties.

            >>> pytta.Default.set_defaults(property1 = value1,
            >>>                            property2 = value2,
            >>>                            propertyN = valueN)

        The default values can be set differently using both declaring
        method, or the set_defaults() function

            >>> pytta.Default.propertyName = propertyValue
            >>> pytta.Default.set_defaults(propertyName = propertyValue)

        """
        for name, value in namevalues.items():  # iterate over the (propertyName = propertyValue) pairs
            try:
                if vars(self)['_'+name] != value:  # Check if user value are different from the ones already set
                    if name in ['device', 'devices']:  # Check if user is changing default audio IO device
                        sd.default.device = value    # If True, changes the sounddevice default audio IO device
                        vars(self)['_'+name] = sd.default.device  # Then loads to PyTTa default device
                    else:
                        vars(self)['_'+name] = value  # otherwise, assign the new value to the desired property
            except KeyError:
                print('You\'ve probably misspelled something.\n' + 'Checkout the property names:\n')
                self.__call__()

    def reset(self):
        """Reset attributes to "factory"."""
        vars(self).clear()
        self.__init__()
        return

    @property
    def samplingRate(self):
        """Sample rate of the signal."""
        return self._samplingRate

    @property
    def lengthDomain(self):
        """
        Information about the recording length.

        May be 'time' or 'samples'.
        """
        return self._lengthDomain

    @property
    def fftDegree(self):
        """
        Adjust the total number of samples to a base 2 number.

            >>> numSamples = 2**fftDegree

        """
        return self._fftDegree

    @property
    def timeLength(self):
        """
        Total time duration of the signal.

            >>> numSamples = samplingRate * timeLength

        """
        return self._timeLength

    @property
    def integration(self):
        """Time interval to integrate for real time analysis."""
        return self._integration

    @property
    def freqMin(self):
        """Minimum frequency."""
        return self._freqMin

    @property
    def freqMax(self):
        """Maximum frequency."""
        return self._freqMax

    @property
    def freqLims(self):
        """Frequency range."""
        return {'min': self._freqMin, 'max': self._freqMax}

    @property
    def device(self):
        """Audio device."""
        return self._device

    @property
    def inChannel(self):
        """List of input channels."""
        return self._inChannel

    @property
    def outChannel(self):
        """List of output channels."""
        return self._outChannel

    @property
    def startMargin(self):
        """Silence duration on signal beginning, in seconds."""
        return self._startMargin

    @property
    def stopMargin(self):
        """Silence duration on signal ending, in seconds."""
        return self._stopMargin

    @property
    def margins(self):
        """Silence duration on signal beginning and ending, in seconds."""
        return {'start': self._startMargin, 'stop': self._stopMargin}

    @property
    def comment(self):
        """Commentary."""
        return self._comment


default = Default()
