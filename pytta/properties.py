# -*- coding: utf-8 -*-
"""
Properties
===========
  
@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br


PyTTa Default Properties:
-------------------------
    
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
    
"""
import sounddevice as sd

__default_device = sd.default.device 
""" Used only to hold the default audio I/O device at pytta import time"""

default = {'samplingRate': 44100,
           'lengthDomain': 'samples',
           'fftDegree': 18,
           'timeLength': 10,
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
    """ 
    Default:
    ========
    
        Holds default parameter values for the <pytta.generate> submodule's functions.
    
    
    Attributes:
    -----------
    
        samplingRate:
            Sampling frequency of the signal;
        lengthDomain:
            Information about the recording length. May be 'time' or 'samples';
        fftDegree:
            Adjusts the total number of samples to a base 2 number (numSamples = 2**fftDegree);
        timeLength:
            Total time duration of the signal (numSamples = samplingRate * timeLength);
        _freqMin:
            Smallest signal frequency of interest;
        _freqMax:
            Greatest signal frequency of interest;
        freqLims['min', 'max']:
            Frequencies of interest bandwidth limits;
        device:
            Devices used for input and output streaming of signals (Measurements only);
        inputChannels:
            Stream input channels of the input device in use (Measurements only);
        outputChannels:
            Stream output channels of the output device in use (Measurements only);
        _startMargin:
            Amount of silence time at signal's beginning (Signals only);
        _stopMargin:
            Amount of silence time at signal's ending (Signals only);
        margins['start', 'stop']:
            Beginning and ending's amount of time left for silence (Signals only);
        comments:
            Any commentary about the signal or measurement the user wants to add.
        
    
    Methods:
    --------
    
        set_defaults(attribute1 = value1, attribute2 = value2, ... , attributeN = valueN):
            Changes attributes values to the ones assigned at the function call. Useful for changing several attributes
            at once.
            
        reset():
            Attributes goes back to "factory default".
    
    """

    _samplingRate = []
    _lengthDomain = []
    _fftDegree = []
    _timeLength = []
    _freqMin = []
    _freqMax = []
    _device = []
    _inChannel = []
    _outChannel = []
    _stopMargin = []
    _startMargin = []
    _comment = []
                        
    def __init__(self):
        """
        Changin "factory" default preferences:
        ======================================
        
            If wanted, the user can set different "factory default" values by changing
            the properties.default dictionary which is used to hold the values that
            the __init__() method loads into the class object at import time
        """

        for name, value in default.items():
            vars(self)['_'+name] = value

    def __setattr__(self,name,value):
        if name in dir(self) and name!= 'device':
            vars(self)['_'+name] = value
        elif name in ['device','devices']:
            self.set_defaults(device = value)
        else:
            raise AttributeError ('There is no default settings for '+repr(name))


    def __call__(self):
        for name, value in vars(self).items():
            if len(name)<=8:
                print(name[1:]+'\t\t =',value)
            else: 
                print(name[1:]+'\t =',value)
                

    def set_defaults(self,**namevalues):
        """
    	Change the values of the "Default" object's properties
    	 
    	>>> pytta.Default.set_defaults(property1 = value1,
    	>>>                            property2 = value2,
    	>>>                            propertyN = valueN)
         
        The default values can be set differently using both declaring method, or
        the set_defaults() function
        
        >>> pytta.Default.propertyName = propertyValue
        >>> pytta.Default.set_defaults(propertyName = propertyValue)
    	 
    	"""
        
        for name, value in namevalues.items(): # iterate over the (propertyName = propertyValue) pairs
            try:
                if vars(self)['_'+name] != value: # Check if user value are different from the ones already set up
                    if name in ['device','devices']: # Check if user is changing default audio IO device
                        sd.default.device = value    # If True, changes the sounddevice default audio IO device
                        vars(self)['_'+name] = sd.default.device # Then loads to PyTTa default device
                    else:
                        vars(self)['_'+name] = value # otherwise, just assign the new value to the desired property
            except KeyError:
                print('You\'ve probably mispelled something.\n' + 'Checkout the property names:\n')
                self.__call__()
    
    def reset(self):
        vars(self).clear()
        self.__init__()

        
    @property
    def samplingRate(self):
        return self._samplingRate

    @property
    def lengthDomain(self):
        return self._lengthDomain
    
    @property
    def fftDegree(self):
        return self._fftDegree
    
    @property
    def timeLength(self):
        return self._timeLength
    
    @property
    def freqMin(self):
        return self._freqMin
    
    @property
    def freqMax(self):
        return self._freqMax
    
    @property
    def freqLims(self):
        return {'min': self._freqMin, 'max': self._freqMax}
    
    @property
    def device(self):
        return self._device
    
    @property
    def inChannel(self):
        return self._inChannel
    
    @property
    def outChannel(self):
        return self._outChannel
    
    @property
    def startMargin(self):
        return self._startMargin
    
    @property
    def stopMargin(self):
        return self._stopMargin
    
    @property
    def margins(self):
        return {'start': self._startMargin, 'stop': self._stopMargin}
    
    @property
    def comment(self):
        return self._comment
    

