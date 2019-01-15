# -*- coding: utf-8 -*-
"""
Properties
===========
  
@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br


PyTTa Default Properties:
    
    As to provide an user friendly signal measurement package, a few default
    values where assigned to the main classes and functions.
    
    These values where set using a dict called "default"
    
    >>> import pytta
    >>> pytta.properties.default
    
    The default values can be set differently using both declaring method, or
    the set_default() function
    
    >>> pytta.properties.default['propertyName'] = propertyValue
    >>> pytta.properties.set_default(propertyName = propertyValue)
    
    The main difference is that using the set_default() function, a list of
    properties can be set at the same time
    
    >>> pytta.properties.set_default(property1 = value1,
    >>>                              property2 = value2,
    >>>                              propertyN = valueN)
    
    The default device start as the one set as default at the user's OS. We
    recommend changing it's value to the desired audio in/out device, as it
    can be identified using list_devices() method
    
    >>> pytta.list_devices()
    
"""
import sounddevice as sd
#import numpy as np

default = {'samplingRate': 44100,
           'fftDegree': 18,
           'timeLength': 10,
           'freqMin': 20,
           'freqMax': 20000,
           'device': [0, 0],
           'inChannel': 1,
           'outChannel': 1,
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
        fftDegree:
            Adjusts the total number of samples to a base 2 number (totalSamples = 2**fftDegree);
        timeLength:
            Total time duration of the signal (totalSamples = samplingRate * timeLength);
        _freqMin:
            Smallest signal frequency of interest;
        _freqMax:
            Greatest signal frequency of interest;
        freqLims['min':'max']:
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
        margins['start','stop']:
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
    _samplingRate = None
    _fftDegree = None
    _timeLength = None
    _freqMin = None
    _freqMax = None
    _device = None
    _inChannel = None
    _outChannel = None
    _stopMargin = None
    _startMargin = None
    _comment = None
                        
    def __init__(self):
        for name, value in default.items():
            vars(self)['_'+name] = value
    
    def __setattr__(self,name,value):
        if name in dir(self) and name!= 'device':
            object.__setattr__(self,'_'+name,value)
        elif name in ['device','devices']:
            self.set_defaults(name,value)
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
    	Change the values of the "default" dictionary
    	 
    	>>> pytta.properties.set_default(property1 = value1,
    	>>>                              property2 = value2,
    	>>>                              propertyN = valueN)
         
        The default values can be set differently using both declaring method, or
        the set_default() function
        
        >>> pytta.properties.default['propertyName'] = propertyValue
        >>> pytta.properties.set_default(propertyName = propertyValue)
    	 
    	"""
         
        for name, value in namevalues.items():
            if vars(self)['_'+name] != value:
                if name in ['device','devices']:
                    sd.default.device = value
                    vars(self)['_'+name] = sd.default.device
                else:
                    vars(self)['_'+name] = value
#        return default
                    
    def reset(self):
        vars(self).clear()
        self.__init__()

        
    @property
    def samplingRate(self):
        return self._samplingRate
    
    @property
    def fftDegree(self):
        return self._fftDegree
    
    @property
    def timeLength(self):
        return self._timeLength
    
    @property
    def freqLims(self):
        return {'min': self._freqMin, 'max': self._freqMax}
    
    @property
    def device(self):
        return self._device
    
    @property
    def inputChannels(self):
        return self._inChannel
    
    @property
    def outputChannels(self):
        return self._outChannel
    
    @property
    def margin(self):
        return {'start': self._startMargin, 'stop':self._stopMargin}
    
    @property
    def comment(self):
        return self._comment
    

