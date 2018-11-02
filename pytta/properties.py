# -*- coding: utf-8 -*-
"""
                     PyTTa
    Object Oriented Python in Technical Acoustics

                    Properties
  
@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br


@Última modificação: 27/10/18

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

default = {'samplingRate': 44100,
           'fftDegree': 18,
           'timeLength': 10,
           'freqMin': 20,
           'winFreqMin':20,
           'freqMax': 20000,
           'winFreqMax': 20000,
           'device': sd.default.device,
           'inch': [1],
           'outch': [1],
           'stopMargin': 0.7,
           'startMargin': 0.3,
           'comment': 'No comments.'}


def set_default(**kargs):
    """
    set_default()
    
        Change the values of the "default" dictionary
        >>> pytta.properties.set_default(property1 = value1,
        >>>                              property2 = value2,
        >>>                              propertyN = valueN)
        
    """
    global default
    for name, value in kargs.items():
        if default[name] != value:
            default[name] = value
    return default

def list_devices():
    """
    list_devices()
    
        Shortcut to sounddevice.query_devices(). Made to exclude the need of
        importing Sounddevice directly just to find out which audio devices can
        be used.
        >>> pytta.list_devices()
        
    """
    return sd.query_devices()
            