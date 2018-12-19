pytt# -*- coding: utf-8 -*-
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
    
    >>> pytta.properties.list_devices()
    
"""
import sounddevice as sd
import numpy as np

default = {'samplingRate': 44100,
           'fftDegree': 18,
           'timeLength': 10,
           'freqMin': 20,
           'freqMax': 20000,
           'device': sd.default.device,
           'inch': np.array([1, 2]),
           'outch': np.array([1, 2]),
           'stopMargin': 0.7,
           'startMargin': 0.3,
           'comment': 'No comments.'}


def set_default(**kargs):
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
    global default
    for name, value in kargs.items():
        if all(default[name]) != value:
            if name == 'device':
                sd.default.device = value
                default[name] = sd.default.device
            elif name=='inch' or name=='outch':
                default[name] = np.array(value)
            else:
                default[name] = value
    return default

