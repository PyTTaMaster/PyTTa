# -*- coding: utf-8 -*-
"""
Classes:
---------

@Autores:
- Matheus Lazarin Alberto, mtslazarin@gmail.com
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

This submodule is mainly the means to an end. PyTTa is made intended to be
user friendly, the manipulation of the classes are documented here, but their
instantiation should be used through the <generate> submodule:

    >>> pytta.generate.sweep()
    >>> pytta.generate.noise()
    >>> pytta.generate.measurement('playrec')
    >>> pytta.generate.measurement('rec', lengthDomain = 'time', timeLen = 5)

This way, the default settings will be loaded into any object instantiated.

User intended classes:

    >>> pytta.SignalObj()
    >>> pytta.RecMeasure()
    >>> pytta.PlayRecMeasure()
    >>> pytta.FRFMeasure()

For further information see the specific class, or method, documentation
"""

# Importing modules
import os
import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.io as sio
import sounddevice as sd
from pytta import default, units
from typing import Optional, List
import time


class PyTTaObj(object):
    """
    PyTTa object class to define some properties and methods to be used
    by any signal and processing classes. pyttaObj is a private class created
    just to shorten attributes declaration to each PyTTa class.

    Properties(self): (default), (dtype), meaning;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * lengthDomain ('time'), (str):
            signal's length domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds for lengthDomain = 'time';

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples for\
            lengthDomain = 'samples';

        * numSamples (samples), (int):
            signal's number of samples

        * freqMin (20), (int):
            minimum frequency bandwidth limit;

        * freqMax (20000), (int):
            maximum frequency bandwidth limit;

        * comment ('No comments.'), (str):
            some commentary about the signal or measurement object;
    """

    def __init__(self,
                 samplingRate=None,
                 freqMin=None,
                 freqMax=None,
                 comment="No comments.",
                 lengthDomain=None,
                 fftDegree=None,
                 timeLength=None,
                 numSamples=None):

        self._lengthDomain = lengthDomain
        self._samplingRate = samplingRate
        self._fftDegree = fftDegree
        self._timeLength = timeLength
        self._numSamples = numSamples
        if freqMin is None or freqMax is None:
            self._freqMin, self._freqMax = default.freqMin, default.freqMax
        else:
            self._freqMin, self._freqMax = freqMin, freqMax
        self._comment = comment
        return

# PyTTaObj Properties
    @property
    def samplingRate(self):
        return self._samplingRate

    @samplingRate.setter
    def samplingRate(self, newSamplingRate):
        self._samplingRate = newSamplingRate
        return

    @property
    def lengthDomain(self):
        return self._lengthDomain

    @lengthDomain.setter
    def lengthDomain(self, newDomain):
        self._lengthDomain = newDomain
        return

    @property
    def fftDegree(self):
        return self._fftDegree

    @fftDegree.setter
    def fftDegree(self, newFftDegree):
        self._fftDegree = newFftDegree
        return

    @property
    def timeLength(self):
        return self._timeLength

    @timeLength.setter
    def timeLength(self, newTimeLength):
        self._timeLength = newTimeLength
        return

    @property
    def numSamples(self):
        return self._numSamples

    @numSamples.setter
    def numSamples(self, newNumSamples):
        self._numSamples = newNumSamples
        return

    @property
    def freqMin(self):
        return self._freqMin

    @freqMin.setter
    def freqMin(self, newFreqMin):
        self._freqMin = round(newFreqMin, 2)
        return

    @property
    def freqMax(self):
        return self._freqMax

    @freqMax.setter
    def freqMax(self, newFreqMax):
        self._freqMax = round(newFreqMax, 2)
        return

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, newComment):
        self._comment = newComment
        return

# PyTTaObj Methods
    def __call__(self):
        for name, value in vars(self).items():
            if len(name) <= 8:
                print(name[1:] + '\t\t =', value)
            else:
                print(name[1:] + '\t =', value)
        return

    def _to_dict(self):
        out = {'samplingRate': self.samplingRate,
               'freqLims': [self.freqMin, self.freqMax],
               'fftDegree': self.fftDegree,
               'lengthDomain': self.lengthDomain,
               'comment': self.comment}
        return out

    def save_mat(self, filename=time.ctime(time.time())):
        myObj = vars(self)
        for key, value in myObj.items():
            if value is None:
                myObj[key] = 0
            if isinstance(myObj[key], dict) and len(value) == 0:
                myObj[key] = 0
        myObjno_ = {}
        for key, value in myObj.items():
            if key.find('_') >= 0:
                key = key.replace('_', '')
            myObjno_[key] = value
        sio.savemat(filename, myObjno_, format='5', oned_as='column')
        return


class CoordinatesObj(object):

    def __init__(self,
                 point=[0, 0, 0],
                 polar=[0, 0, 0],
                 ref='Your eyes',
                 unit='m'):
        self.point = point
        if self.point == [0, 0, 0]:
            self.polar = polar
        self.ref = ref
        self.unit = unit

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'point={self.point!r}, '
                f'polar={self.polar!r}, '
                f'ref={self.ref!r}, '
                f'unit={self.unit!r})')

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, newpoint):
        if isinstance(newpoint, list) and len(newpoint) == 3:
            self._point = newpoint
            # Calc polar coord
            r = (self._point[0]**2+self._point[1]**2+self._point[2]**2)**(1/2)
            if r != 0:
                elev = np.arccos(self._point[2]/r)
                azi = np.arctan(self._point[1] /
                                self._point[0])
            else:
                elev = 0
                azi = 0
            self._polar = [r, elev, azi]
        else:
            TypeError('Cartesian three-dimensional coordinates must be a list,\
                       e.g. [X, Y, Z])')

    @property
    def polar(self):
        polarInDeg = [self._polar[0],
                      self._polar[1]/np.pi*180,
                      self._polar[2]/np.pi*180]
        return polarInDeg

    @polar.setter
    def polar(self, newpolar):
        if isinstance(newpolar, list) and len(newpolar) == 3:
            self._polar = [newpolar[0],
                           newpolar[1]/180*np.pi,
                           newpolar[2]/180*np.pi]
            # Calc cartesian coord
            x = self._polar[0] * np.sin(self._polar[1]) * \
                np.cos(self._polar[2])
            y = self._polar[0] * np.sin(self._polar[1]) * \
                np.sin(self._polar[2])
            z = self._polar[0] * np.cos(self._polar[1])
            self._point = [x, y, z]
        else:
            TypeError('Polar three-dimensional coordinates must be a list,\
                       e.g. [Radius, Elevation, Azimuth])')

    @property
    def ref(self):
        return self._ref

    @ref.setter
    def ref(self, newref):
        if isinstance(newref, str):
            self._ref = newref
        else:
            TypeError('ref must be a string,\
                       e.g. \'Room inferior back left corner\'')

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, newunit):
        if isinstance(newunit, str):
            self._unit = newunit
        else:
            TypeError('unit must be a string,\
                       e.g. \'m\'')

    def _to_dict(self):
        out = {'point': self.point,
               'ref': self.ref,
               'unit': self.unit}
        return out


class ChannelObj(object):
    """
    Base class for signal meta information about the IO channel it's been
    acquired

    Parameters and Attributes:
    ----------------------------
        Every parameter becomes the homonim attribute.

        .. attribute:: name:
            String with name or ID;

        .. attribute:: unit:
            String with International System units for the data, e.g. 'Pa', \
            'V', 'FS';

        .. attribute:: CF:
            Calibration factor, numerically convert normalized float32 values \
            to :attr:`unit` values;

        .. attribute:: calibCheck:
            :type:`bool`, information about wether :attr:`CF` is applied \
            (True), or not (False -> default);

    Special methods:
    ------------------

        .. method:: __mul__:
            perform :attr:`unit` concatenation  # TODO unit conversion.

        .. method:: __truediv__:
            perform :attr:`unit` concatenation  # TODO unit conversion.

    """
    def __init__(self, num, name=None, unit='FS', CF=1, calibCheck=False,
                 coordinates=CoordinatesObj(), orientation=CoordinatesObj()):
        self.num = num
        if name is None:
            self.name = 'Ch. '+str(self.num)
        else:
            self.name = name
        self.unit = unit
        self.CF = CF
        self.calibCheck = calibCheck
        self.coordinates = coordinates
        self.orientation = orientation

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num={self.num!r}, '
                f'name={self.name!r}, '
                f'unit={self.unit!r}, '
                f'CF={self.CF!r}, '
                f'calibCheck={self.calibCheck!r}, '
                f'coordinates={self.coordinates.point!r}, '
                f'orientation={self.orientation.point!r})')

    def __mul__(self, other):
        if not isinstance(other, ChannelObj):
            raise TypeError('Can\'t "multiply" by other \
                            type than a ChannelObj')
        newCh = ChannelObj(self.num, name=self.name+'.'+other.name,
                           unit=self.unit+'.'+other.unit,
                           CF=self.CF*other.CF,
                           calibCheck=self.calibCheck if self.calibCheck
                           else other.calibCheck)
        return newCh

    def __truediv__(self, other):
        if not isinstance(other, ChannelObj):
            raise TypeError('Can\'t "divide" by other type than a ChannelObj')
        if self.unit == other.unit:
            newunit = 'FS'
        else:
            newunit = self.unit+'/'+other.unit
        newCh = ChannelObj(self.num,
                           # name=self.name+'/'+other.name,
                           unit=newunit,
                           CF=self.CF/other.CF,
                           calibCheck=self.calibCheck if self.calibCheck
                           else other.calibCheck)
        return newCh

    def calib_volt(self, refSignalObj, refVrms, refFreq):
        Vrms = np.max(np.abs(refSignalObj.freqSignal[:, 0])) / (2**(1/2))
        print(Vrms)
        freqFound = np.round(
                refSignalObj.freqVector[np.where(
                        np.abs(refSignalObj.freqSignal)
                        == np.max(np.abs(refSignalObj.freqSignal)))[0]])
        if freqFound != refFreq:
            print('\x1b[0;30;43mATENTTION! Found calibration frequency ('
                  + '{:.2}'.format(freqFound)
                  + ' [Hz]) differs from refFreq ('
                  + '{:.2}'.format(refFreq) + ' [Hz])\x1b[0m')
        self.CF = refVrms/Vrms
        self.unit = 'V'
        return

    def calib_press(self, refSignalObj, refPrms, refFreq):
        Prms = np.max(np.abs(refSignalObj.freqSignal[:, 0])) / (2**(1/2))
        print(Prms)
        freqFound = np.round(refSignalObj.freqVector[np.where(
                np.abs(refSignalObj.freqSignal)
                == np.max(np.abs(refSignalObj.freqSignal)))[0]])
        if freqFound != refFreq:
            print('\x1b[0;30;43mATENTTION! Found calibration frequency ('
                  + '{:.2}'.format(freqFound)
                  + ' [Hz]) differs from refFreq ('
                  + '{:.2}'.format(refFreq) + ' [Hz])\x1b[0m')
        self.CF = refPrms/Prms
        self.unit = 'Pa'
        return

# ChannelObj properties
    @property
    def num(self):
        return self._num

    @num.setter
    def num(self, ber):
        if type(ber) is not int:
            try:
                ber = int(ber)
            except ValueError:
                raise TypeError("Channel number must be an integer.")
        elif ber < 1:
            raise ValueError("Channel number must be greater than 1.")
        self._num = ber
        return

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, newname):
        if isinstance(newname, str):
            self._name = newname
        else:
            raise TypeError('Channel name must be a string.')
        return

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, newunit):
        if isinstance(newunit, str):
            if newunit in units:
                self._unit = newunit
                self.dBName = units[newunit][0]
                self.dBRef = units[newunit][1]
            else:
                self._unit = newunit
                self.dBName = 'dB'
                self.dBRef = 1
        else:
            raise TypeError('Channel unit must be a string.')
        return

    @property
    def CF(self):
        return self._CF

    @CF.setter
    def CF(self, newCF):
        if isinstance(newCF, float) or isinstance(newCF, int):
            self._CF = newCF
        else:
            raise TypeError('Channel correction factor must be a number.')
        return

    @property
    def calibCheck(self):
        return self._calibCheck

    @calibCheck.setter
    def calibCheck(self, newcalibCheck):
        if isinstance(newcalibCheck, bool):
            self._calibCheck = newcalibCheck
        else:
            raise TypeError('Channel calibration check must be True or False.')
        return

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, newcoord):
        if isinstance(newcoord, list) and len(newcoord) == 3:
            self._coordinates.point = newcoord
        elif isinstance(newcoord, CoordinatesObj):
            self._coordinates = newcoord
        else:
            raise TypeError('Coordinates must be a list with the ' +
                            'three-dimensional cartesian points (e.g. [3, 3,' +
                            ' 4]), or a CoordinatesObj.')

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, neworient):
        if isinstance(neworient, list) and len(neworient) == 3:
            self._orientation.point = neworient
        elif isinstance(neworient, CoordinatesObj):
            self._orientation = neworient
        else:
            raise TypeError('Orientation must be a list with the ' +
                            'three-dimensional cartesian points (e.g. [3, 3,' +
                            ' 4]), or a CoordinatesObj.')
        return

    def _to_dict(self):
        out = {'calib': [self.CF, self.calibCheck],
               'unit': self.unit,
               'name': self.name,
               'coordinates': self.coordinates._to_dict(),
               'orientation': self.orientation._to_dict()}
        return out


class ChannelsList(object):
    """
    .. class:: ChannelsList(self, chN=0):

        Class to wrap a list of ChannelObj and handle multi-channel SignalObj \
        operations.

        :param int chN: Number of initialized ChannelObj inside the list;

        .. attribute:: _channels: List holding each ChannelObj;

        # TODO rest of it

    Attributes:
    ------------

        .. attribute:: _channels:
            :type:`list` holding each :class:`ChannelObj`
    """
    def __init__(self, chList=None):
        self._channels = []
        if chList is not None:
            if type(chList) is list:
                for memb in chList:
                    if type(memb) is ChannelObj:
                        self._channels.append(memb)
                    else:
                        self._channels.append(ChannelObj(memb))
            elif type(chList) is int:
                self._channels.append(ChannelObj(chList))
            elif type(chList) is ChannelObj:
                self._channels.append(chList)
            else:
                raise TypeError('List initializer must be either positive int,\
                                ChannelObj, a list of positive int or\
                                ChannelObj.')
        else:
            self._channels.append(ChannelObj(1))
        return

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self._channels!r})')

    def __len__(self):
        return len(self._channels)

    def __getitem__(self, key):
        try:
            return self._channels[key]
        except IndexError:
            raise IndexError("Out of range.")

    def __setitem__(self, key, item):
        try:
            self._channels[key] = item
            return
        except IndexError:
            raise IndexError("Out of range.")

    def __mul__(self, otherList):
        if not isinstance(otherList, ChannelsList):
            raise TypeError('Can\'t "multiply" by other \
                            type than a ChannelsList')
        if len(self) > 1:
            if len(otherList) > 1:
                if len(self) != len(otherList):
                    raise ValueError("Both ChannelsList-like objects must \
                                     have the same number of channels.")
                newChList = ChannelsList([self[index]*otherList[index]
                                          for index in range(len(self))])
            else:
                newChList = ChannelsList([self[index] * otherList[0]
                                          for index in range(len(self))])
        else:
            if len(otherList) > 1:
                newChList = ChannelsList([self[0]*otherList[index]
                                          for index in range(len(otherList))])
            else:
                newChList = ChannelsList([self[0]*otherList[0]])
        return newChList

    def __truediv__(self, otherList):
        if not isinstance(otherList, ChannelsList):
            raise TypeError('Can\'t "divide" by other \
                            type than a ChannelsList')
        if len(self) > 1:
            if len(otherList) > 1:
                if len(self) != len(otherList):
                    raise ValueError("Both ChannelsList-like objects must \
                                     have the same number of channels.")
                newChList = ChannelsList([self[index]/otherList[index]
                                          for index in range(len(self))])
            else:
                newChList = ChannelsList([self[index] / otherList[0]
                                          for index in range(len(self))])
        else:
            if len(otherList) > 1:
                newChList = ChannelsList([self[0]/otherList[index]
                                          for index in range(len(otherList))])
            else:
                newChList = ChannelsList([self[0]/otherList[0]])
        return newChList

    def mapping(self):
        out = []
        for obj in self._channels:
            out.append(obj.num)
        return out

    def CFlist(self):
        out = []
        for obj in self._channels:
            out.append(obj.CF)
        return out

    def _to_dict(self):
        out = {}
        for ch in self._channels:
            out[ch.num] = ch._to_dict()
        return out

    def append(self, newCh):
        if isinstance(newCh, ChannelObj):
            self._channels.append(newCh)
        return

    def pop(self, Ch):
        if Ch not in range(len(self)):
            raise IndexError('Inexistent Channel index')
        self._channels.pop(Ch)
        return

    def conform_to(self, rule=None):
        if isinstance(rule, SignalObj):
            dCh = rule.num_channels() - len(self)
            # Adjusting number of channels
            if dCh > 0:
                newIndex = len(self)
                for i in range(dCh):
                    self.append(ChannelObj(num=(newIndex+i+1)))
            if dCh < 0:
                for i in range(0, -dCh):
                    self._channels.pop(-1)
        elif isinstance(rule, list):
            self._channels = []
            for index in rule:
                self.append(ChannelObj(num=index+1, name='Channel ' +
                                       str(index)))
        elif rule is None:
            count = 1
            newchs = []
            # Adjusting channel's numbers
            for ch in self._channels:
                newchs.append(ChannelObj(num=count, name=ch.name, unit=ch.unit,
                                         CF=ch.CF, calibCheck=ch.calibCheck,
                                         coordinates=ch.coordinates,
                                         orientation=ch.orientation))
                count += 1
            # Adjusting channel's names
            for idx1 in range(len(newchs)):
                neq = 2
                for idx2 in range(len(newchs)):
                    if idx1 != idx2:
                        if newchs[idx1].name == newchs[idx2].name:
                            newchs[idx2].name = newchs[idx1].name + \
                                ' - ' + str(neq)
                            neq += 1
            self._channels = newchs
        else:
            raise TypeError('Rule must be an SignalObj or a list with ' +
                            'channel\'s numbers')
        return

    def rename_channels(self):
        for chIndex in range(len(self)):
            newname = 'Ch. ' + str(chIndex+1)
            self._channels[chIndex].name = newname
        return


class SignalObj(PyTTaObj):
    """
    Signal object class.

    Creation parameters:
    ------------

        * signalArray (ndarray | list), (NumPy array):
            signal at specified domain

        * domain ('time'), (str):
            domain of the input array;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * freqMin (20), (int):
            minimum frequency bandwidth limit;

        * freqMax (20000), (int):
            maximum frequency bandwidth limit;

        * comment ('No comments.'), (str):
            some commentary about the signal or measurement object;


    Attributes:
    ------------

        * timeSignal (ndarray), (NumPy array):
            signal at time domain;

        * timeVector (ndarray), (NumPy array):
            time reference vector for timeSignal;

        * freqSignal (ndarray), (NumPy array):
            signal at frequency domain;

        * freqVector (ndarray), (NumPy array):
            frequency reference vector for freqSignal;

        * unit (None), (str):
            signal's unit. May be 'V' or 'Pa';

        * channelName (dict), (dict/str):
            channels name dict;

        * lengthDomain ('time'), (str):
            input array's domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's duration;

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples;

        * numSamples (samples), (int):
            signal's number of samples


    Methods:
    ---------

        * num_channels():
            return the number of channels in the instace;

        * max_level():
            return the channel's max levels;

        * play():
            reproduce the timeSignal with default output device;

        * plot_time():
            generates the signal's historic graphic;

        * plot_freq():
            generates the signal's spectre graphic;

        * calib_voltage(refSignalObj,refVrms,refFreq):
            voltage calibration from an input SignalObj;

        * calib_pressure(refSignalObj,refPrms,refFreq):
            pressure calibration from an input SignalObj;

        * save_mat(filename):
            save a SignalObj to a .mat file;
    """

    def __init__(self,
                 signalArray=np.array([0], ndmin=2).T,
                 domain='time',
                 *args,
                 **kwargs):
        # Converting signalArray from list to np.array
        if isinstance(signalArray, list):
            signalArray = np.array(signalArray)
        # Checking input array dimensions
        if self.size_check(signalArray) > 2:
            message = "No 'pyttaObj' is able handle to arrays with more \
                        than 2 dimensions, '[:,:]', YET!."
            raise AttributeError(message)
        elif self.size_check(signalArray) == 1:
            signalArray = np.array(signalArray, ndmin=2)
        if signalArray.shape[1] > signalArray.shape[0]:
            signalArray = signalArray.T
        super().__init__(*args, **kwargs)

        self.channels = ChannelsList()
        self.lengthDomain = domain
        if self.lengthDomain == 'freq':
            self.freqSignal = signalArray  # [-] signal in frequency domain
        elif self.lengthDomain == 'time':
            self.timeSignal = signalArray  # [-] signal in time domain
        else:
            self.timeSignal = signalArray
            self.lengthDomain = 'time'
            print('Taking the input as a time domain signal')

# SignalObj Properties
    @property
    def timeVector(self):
        return self._timeVector

    @property
    def freqVector(self):
        return self._freqVector

    @property
    def timeSignal(self):
        return self._timeSignal

    @timeSignal.setter
    def timeSignal(self, newSignal):
        if isinstance(newSignal, np.ndarray):
            if self.size_check(newSignal) == 1:
                newSignal = np.array(newSignal, ndmin=2)
            if newSignal.shape[1] > newSignal.shape[0]:
                newSignal = newSignal.T
            self._timeSignal = np.array(newSignal)
            self._freqSignal = np.fft.rfft(self._timeSignal, axis=0, norm=None)
            self._freqSignal = 1/len(self._freqSignal)*self._freqSignal
            # number of samples
            self._numSamples = len(self._timeSignal)
            # size parameter
            self._fftDegree = np.log2(self._numSamples)
            # duration in [s]
            self._timeLength = self.numSamples/self.samplingRate
            # [s] time vector (x axis)
            self._timeVector = np.linspace(0, self.timeLength -
                                           (1/self.samplingRate),
                                           self.numSamples)
            # [Hz] frequency vector (x axis)
            self._freqVector = np.linspace(0, (self.numSamples - 1) *
                                           self.samplingRate /
                                           (2*self.numSamples),
                                           (self.numSamples/2)+1
                                           if self.numSamples % 2 == 0
                                           else (self.numSamples+1)/2)
            self.channels.conform_to(self)
        else:
            raise TypeError('Input array must be a numpy ndarray')
        return

    @property
    def freqSignal(self):
        return self._freqSignal

    @freqSignal.setter
    def freqSignal(self, newSignal):
        if isinstance(newSignal, np.ndarray):
            if self.size_check(newSignal) == 1:
                newSignal = np.array(newSignal, ndmin=2)
            if newSignal.shape[1] > newSignal.shape[0]:
                newSignal = newSignal.T
            self._freqSignal = np.array(newSignal)
            self._timeSignal = np.fft.irfft(self._freqSignal,
                                            axis=0, norm=None)
            self._numSamples = len(self.timeSignal)  # [-] number of samples
            self._fftDegree = np.log2(self.numSamples)  # [-] size parameter
            self._timeLength = self.numSamples/self.samplingRate
            self._timeVector = np.arange(0, self.timeLength,
                                         1/self.samplingRate)
            self._freqVector = np.linspace(0, (self.numSamples-1) *
                                           self.samplingRate /
                                           (2*self.numSamples),
                                           (self.numSamples/2) + 1
                                           if self.numSamples % 2 == 0
                                           else (self.numSamples+1)/2)
            self.channels.conform_to(self)
        else:
            raise TypeError('Input array must be a numpy ndarray')
        return

    @property
    def coordinates(self):
        coords = []
        for chIndex in range(self.num_channels()):
            coords.append(self.channels[chIndex].coordinates)
        return coords

    @property
    def orientation(self):
        orientations = []
        for chIndex in range(self.num_channels()):
            orientations.append(self.channels[chIndex].orientation)
        return orientations

# SignalObj Methods
    def mean(self):
        return SignalObj(signalArray=np.mean(self.timeSignal, 1),
                         lengthDomain='time', samplingRate=self.samplingRate)

    @property
    def numChannels(self):
        return self.num_channels()

    def num_channels(self):
        try:
            numChannels = self.timeSignal.shape[1]
        except IndexError:
            numChannels = 1
        return numChannels

    def max_level(self):
        maxlvl = []
        for chIndex in range(self.num_channels()):
            maxAmplitude = np.max(np.abs(self.timeSignal[:, chIndex]))
            maxlvl.append(20*np.log10(maxAmplitude /
                                      self.channels[chIndex].dBRef))
        return maxlvl

    def size_check(self, inputArray=[]):
        if inputArray == []:
            inputArray = self.timeSignal[:]
        return np.size(inputArray.shape)

    def play(self, outChannel=None, latency='low', **kwargs):
        """
        Play method
        """
        if outChannel is None:
            if self.num_channels() <= 1:
                outChannel = default.outChannel
            elif self.num_channels() > 1:
                outChannel = np.arange(1, self.num_channels()+1)
        sd.play(self.timeSignal, self.samplingRate,
                mapping=outChannel, **kwargs)
        return

    def plot_time(self):
        """
        Time domain plotting method
        """
        # DB
        plt.figure(figsize=(10, 5))
        if self.num_channels() > 1:
            for chIndex in range(self.num_channels()):
                label = self.channels[chIndex].name +\
                        ' [' + self.channels[chIndex].unit + ']'
                plt.plot(self.timeVector,
                         self.timeSignal[:, chIndex], label=label)
        else:
            chIndex = 0
            label = self.channels[chIndex].name +\
                ' [' + self.channels[chIndex].unit + ']'
            plt.plot(self.timeVector,
                     self.timeSignal[:, chIndex], label=label)
        plt.legend(loc='best')
        plt.grid(color='gray', linestyle='-.', linewidth=0.4)
        plt.axis((self.timeVector[0] - 10/self.samplingRate,
                  self.timeVector[-1] + 10/self.samplingRate,
                  1.05 * np.min(self.timeSignal),
                  1.05 * np.max(self.timeSignal)))
        plt.xlabel(r'$Time$ [s]')
        plt.ylabel(r'$Amplitude$')
        return

    def plot_freq(self, smooth=False):
        """
        Frequency domain dB plotting method
        """
        plt.figure(figsize=(10, 5))
        if self.num_channels() > 1:
            for chIndex in range(0, self.num_channels()):
                if smooth:
                    Signal = ss.savgol_filter(np.squeeze(np.abs(
                             self.freqSignal[:, chIndex]) / (2**(1/2))),
                             31, 3)
                else:
                    Signal = self.freqSignal[:, chIndex] / (2**(1/2))
                dBSignal = 20 * np.log10(np.abs(Signal)
                                         / self.channels[chIndex].dBRef)
                label = self.channels[chIndex].name \
                    + ' [' + self.channels[chIndex].dBName + ' ref.: ' \
                    + str(self.channels[chIndex].dBRef) + ' ' \
                    + self.channels[chIndex].unit + ']'
                plt.semilogx(self.freqVector, dBSignal, label=label)
        else:
            chIndex = 0
            if smooth:
                Signal = ss.savgol_filter(np.squeeze(np.abs(
                         self.freqSignal[:, chIndex]) / (2**(1/2))),
                         31, 3)
            else:
                Signal = self.freqSignal[:, chIndex] / (2**(1/2))
            dBSignal = 20 * np.log10(np.abs(Signal)
                                     / self.channels[chIndex].dBRef)
            label = self.channels[chIndex].name + ' ['\
                + self.channels[chIndex].dBName + ' ref.: '\
                + str(self.channels[chIndex].dBRef) + ' '\
                + self.channels[chIndex].unit + ']'
            plt.semilogx(self.freqVector, dBSignal, label=label)
        plt.grid(color='gray', linestyle='-.', linewidth=0.4)
        plt.legend(loc='best')
        if np.max(dBSignal) > 0:
            ylim = [1.05*np.min(dBSignal), 1.12*np.max(dBSignal)]
        else:
            ylim = [np.min(dBSignal) - 2, np.max(dBSignal) + 2]
        plt.axis((self.freqMin, self.freqMax, ylim[0], ylim[1]))
        plt.xlabel(r'$Frequency$ [Hz]')
        plt.ylabel(r'$Magnitude$ in dB')
        return

    def plot_spectrogram(self, window='hann', winSize=1024, overlap=0.5):
        _spectrogram, _specTime, _specFreq\
            = self._calc_spectrogram(self.timeSignal[:, 0], overlap,
                                     window, winSize)
        plt.pcolormesh(_specTime.T, _specFreq.T, _spectrogram,
                       cmap=plt.jet(), vmin=-120)
        plt.xlabel(r'$Time$ [s]')
        plt.ylabel(r'$Frequency$ [Hz]')
        plt.colorbar()
        return

    def calib_voltage(self, chIndex, refSignalObj, refVrms=1, refFreq=1000):
        """
        calibVoltage method: use informed SignalObj with a calibration voltage
        signal, and the reference RMS voltage to calculate the Correction
        Factor.

            >>> SignalObj.calibVoltage(chIndex,refSignalObj,refVrms,refFreq)

        Parameters:
        ------------

            * chIndex (), (int):
                channel index for calibration. Starts in 0;

            * refSignalObj (), (SignalObj):
                SignalObj with the calibration recorded signal;

            * refVrms (1.00), (float):
                the reference voltage provided by the voltage calibrator;

            * refFreq (1000), (int):
                the reference sine frequency provided by the voltage
                calibrator;
        """
        if chIndex in range(self.num_channels()):
            self.channels[chIndex].calib_volt(refSignalObj, refVrms, refFreq)
#            Vrms = np.max(np.abs(refSignalObj.freqSignal[:, 0])) / (2**(1/2))
#            print(Vrms)
#            freqFound = np.round(
#                    refSignalObj.freqVector[np.where(
#                            np.abs(refSignalObj.freqSignal)
#                            == np.max(np.abs(refSignalObj.freqSignal)))[0]])
#            if freqFound != refFreq:
#                print('\x1b[0;30;43mATENTTION! Found calibration frequency ('
#                      + '{:.2}'.format(freqFound)
#                      + ' [Hz]) differs from refFreq ('
#                      + '{:.2}'.format(refFreq) + ' [Hz])\x1b[0m')
#            self.channels[chIndex].CF = refVrms/Vrms
#            self.channels[chIndex].unit = 'V'
            self.timeSignal[:, chIndex] = self.timeSignal[:, chIndex]\
                * self.channels[chIndex].CF
            self.channels[chIndex].calibCheck = True
        else:
            raise IndexError('chIndex greater than channels number')
        return

    def calib_pressure(self, chIndex, refSignalObj,
                       refPrms=1.00, refFreq=1000):
        """
        calibPressure method: use informed SignalObj, with a calibration
        acoustic pressure signal, and the reference RMS acoustic pressure to
        calculate the Correction Factor.

            >>> SignalObj.calibPressure(chIndex,refSignalObj,refPrms,refFreq)

        Parameters:
        -------------

            * chIndex (), (int):
                channel index for calibration. Starts in 0;

            * refSignalObj (), (SignalObj):
                SignalObj with the calibration recorded signal;

            * refPrms (1.00), (float):
                the reference pressure provided by the acoustic calibrator;

            * refFreq (1000), (int):
                the reference sine frequency provided by the acoustic
                calibrator;
        """

        if chIndex in range(self.num_channels()):
            self.channels[chIndex].calib_press(refSignalObj, refPrms, refFreq)
#            Prms = np.max(np.abs(refSignalObj.freqSignal[:, 0])) / (2**(1/2))
#            print(Prms)
#            freqFound = np.round(refSignalObj.freqVector[np.where(
#                    np.abs(refSignalObj.freqSignal)
#                    == np.max(np.abs(refSignalObj.freqSignal)))[0]])
#            if freqFound != refFreq:
#                print('\x1b[0;30;43mATENTTION! Found calibration frequency ('
#                      + '{:.2}'.format(freqFound)
#                      + ' [Hz]) differs from refFreq ('
#                      + '{:.2}'.format(refFreq) + ' [Hz])\x1b[0m')
#            self.channels[chIndex].CF = refPrms/Prms
#            self.channels[chIndex].unit = 'Pa'
            self.timeSignal[:, chIndex] = self.timeSignal[:, chIndex]\
                * self.channels[chIndex].CF
            self.channels[chIndex].calibCheck = True
        else:
            raise IndexError('chIndex greater than channels number')
        return

    def save(self, dirname=time.ctime(time.time())):
        mySigObj = self._to_dict()
        with zipfile.ZipFile(dirname + '.pytta', 'w') as zdir:
            filename = 'timeSignal.mat'
            with open(filename, 'wb+'):
                sio.savemat(filename, mySigObj['timeSignalAddress'],
                            do_compression=True,
                            format='5', oned_as='column')
            zdir.write(filename, compress_type=zipfile.ZIP_DEFLATED)
            os.remove(filename)
            mySigObj['timeSignalAddress'] = filename
            filename = 'SignalObj.json'
            with open(filename, 'w') as f:
                json.dump(mySigObj, f, indent=4)
            zdir.write(filename, compress_type=zipfile.ZIP_DEFLATED)
            os.remove(filename)
        return dirname + '.pytta'

    def _to_dict(self):
        out = super()._to_dict()
        out['channels'] = self.channels._to_dict()
        out['timeSignalAddress'] = {'timeSignal': self.timeSignal[:]}
        return out

    def __truediv__(self, other):
        """
        Frequency domain division method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike.")
        if other.samplingRate != self.samplingRate:
            raise TypeError("Both SignalObj must have the same sampling rate.")
        result = SignalObj(np.zeros(self.timeSignal.shape),
                           samplingRate=self.samplingRate)
        result.channels = self.channels
        if self.num_channels() > 1:
            if other.num_channels() > 1:
                if other.num_channels() != self.num_channels():
                    raise ValueError("Both signal-like objects must have the \
                                     same number of channels.")
                result_freqSignal = np.zeros(self.freqSignal.shape,
                                             dtype=np.complex_)
                for channel in range(other.num_channels()):
                    result.freqSignal[:, channel] = \
                        self.freqSignal[:, channel] \
                        / other.freqSignal[:, channel]
                result.freqSignal = result_freqSignal
            else:
                result_freqSignal = np.zeros(self.freqSignal.shape,
                                             dtype=np.complex_)
                for channel in range(self.num_channels()):
                    result_freqSignal[:, channel] = \
                        self.freqSignal[:, channel] \
                        / other.freqSignal[:, 0]
                result.freqSignal = result_freqSignal
        else:
            result.freqSignal = self.freqSignal / other.freqSignal
        result.channels = self.channels / other.channels
        return result

    def __add__(self, other):
        """
        Time domain addition method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike.")
        if other.samplingRate != self.samplingRate:
            raise TypeError("Both SignalObj must have the same sampling rate.")
        result = SignalObj(samplingRate=self.samplingRate)
        result.domain = 'time'
        if self.size_check() > 1:
            if other.size_check() > 1:
                if other.size_check() != self.size_check():
                    raise ValueError("Both signal-like objects must have\
                                     the same number of channels.")
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:, channel]\
                        + other._timeSignal[:, channel]
            else:
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:, channel]\
                        + other._timeSignal
        else:
            result.timeSignal = self._timeSignal + other._timeSignal
        return result

    def __sub__(self, other):
        """
        Time domain subtraction method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike.")
        if other.samplingRate != self.samplingRate:
            raise TypeError("Both SignalObj must have the same sampling rate.")
        result = SignalObj(samplingRate=self.samplingRate)
        result.domain = 'time'
        if self.size_check() > 1:
            if other.size_check() > 1:
                if other.size_check() != self.size_check():
                    raise ValueError("Both signal-like objects must have\
                                     the same number of channels.")
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:, channel]\
                        - other._timeSignal[:, channel]
            else:
                for channel in range(other.num_channels()):
                    result.timeSignal = self._timeSignal[:, channel]\
                        - other._timeSignal
        else:
            result.timeSignal = self._timeSignal - other._timeSignal
        return result

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'SignalArray=ndarray, domain={self.lengthDomain!r}, '
                f'samplingRate={self.samplingRate!r}, '
                f'freqMin={self.freqMin!r}, '
                f'freqMax={self.freqMax!r}, '
                f'comment={self.comment!r})')

    def _calc_spectrogram(self, timeData=None, overlap=0.5,
                          winType='hann', winSize=1024):
        if timeData is None:
            timeData = self.timeSignal
            if self.num_channels() > 1:
                timeData = timeData[:, 0]
        window = eval('ss.windows.' + winType)(winSize)
        nextIdx = int(winSize*overlap)
        rng = int(timeData.shape[0]/winSize/overlap - 1)
        _spectrogram = np.zeros((winSize//2 + 1, rng))
        _specFreq = np.linspace(0, self.samplingRate//2, winSize//2 + 1)
        _specTime = np.linspace(0, self.timeVector[-1], rng)
        for N in range(rng):
            try:
                strIdx = N*nextIdx
                endIdx = winSize + N*nextIdx
                sliceAudio = window*timeData[strIdx:endIdx]
                sliceFFT = np.fft.rfft(sliceAudio, axis=0)
                sliceMag = np.absolute(sliceFFT) * (2/sliceFFT.size)
                _spectrogram[:, N] = 20*np.log10(sliceMag)
            except IndexError:
                sliceAudio = timeData[-winSize:]
                sliceFFT = np.fft.rfft(sliceAudio, axis=0)
                sliceMag = np.absolute(sliceFFT) * (2/sliceFFT.size)
                _spectrogram[:, N] = 20*np.log10(sliceMag)
        return _spectrogram, _specTime, _specFreq


# ImpulsiveResponse class
class ImpulsiveResponse(PyTTaObj):
    """
        This class is a container of SignalObj, intended to provide a system's
        impulsive response along with the excitation signal and the recorded
        signal used to compute the response.

        The access to this class is (TODO) provided by the function:

            >>> pytta.get_IR( excitation (SignalOjb), recording (SignalOjb),
                              coordinates (dict), method (str),
                              winType (str | tuple), winSize (int),
                              overlap (float))

        And as an output of the FRFMeasure.run() method:

            >>> myMeas = pytta.generate.measurement('frf')
            >>> myIR = myMeas.run()
            >>> type(myIR)
            classes.ImpulsiveResponse

        The parameter passed down to the function are the same that initialize
        the class, and are explained as follows:

        Creation parameters:
        ---------------------

            * excitation (SignalObj):
                The signal-like object used as excitation signal on the
                measurement-like object;

            * recording (SignalObj):
                the recorded signal-like object, obtained directly from the
                audio interface used on the measurement-like object;

            * coordinates (dict):
                A dict that contains the following keys:

                    * points (list):
                        A list handled by the get_channels_points(ch) and
                        set_channels_points(ch, pt) object methods. Must be
                        organized as [ [x1, y1, z1], [x2, y2, z2], ...] with
                        x, y and z standing for the distance from the
                        reference point;

                    * reference (str):
                        A short description of a place that is considered the
                        system origin, e.g. 'south-east-floor corner';

                    * unit (str):
                        The unit in which the points values are taken,
                        e.g. 'm';

            * method (str):
                The way that the impulsive response should be computed, accepts
                "linear", "H1", "H2" and "Ht" as values:

                    * "linear":
                        Computes using the spectral division of the signals;

                    * "H1":
                        Uses power spectral density Ser divided by See, with
                        "e" standing for "excitation" and "r" for "recording;

                    * "H2":
                        uses power spectral density Srr divided by Sre, with
                        "e" standing for "excitation" and "r" for "recording;

                    * "Ht":
                        uses the formula: TODO;

            * winType (str | tuple) (optional):
                The name of the window used by the scipy.signal.csd function
                to compute the power spectral density, (only for method="H1",
                method="H2" and method="Ht"). The possible values are:

                    >>> boxcar, triang, blackman, hamming, hann, bartlett,
                        flattop, parzen, bohman, blackmanharris, nuttall,
                        barthann, kaiser (needs beta), gaussian (needs standard
                        deviation), general_gaussian (needs power, width),
                        slepian (needs width), dpss (needs normalized half-
                        bandwidth), chebwin (needs attenuation), exponential
                        (needs decay scale), tukey (needs taper fraction).

                If the window requires no parameters, then window can be
                a string.

                If the window requires parameters, then window must be a tuple
                with the first argument the string name of the window, and the
                next arguments the needed parameters.

                    source:
                        https://docs.scipy.org/doc/scipy/reference/generated\
                        /scipy.signal.csd.html

            * winSize (int) (optional):
                The size of the window used by the scipy.signal.csd function
                to compute the power spectral density, (only for method="H1",
                method="H2" and method="Ht");

            * overlap (float) (optional):
                the overlap ratio of the window used by the scipy.signal.csd
                function to compute the power spectral density, (only for
                method ="H1", method="H2" and method="Ht").


        The class's attribute are described next:

        Attributes:
        ------------

            * excitation | inputSignal:
                Both names are valid, returns the excitation signal given as
                parameter at the object instantiation;

            * recording | outputSignal:
                Both names are valid, returns the recording signal given as
                parameter at the object instantiation;

            * irSignal | IR | tfSignal | TF | systemSignal:
                All names are valid, returns the computed impulsive response
                signal-like object;

            * coordinates:
                Returns the coordinates parameter passed at the object instan-
                tiation. It's "points" values may be updated;

            * methodInfo:
                Returns a dict with the "method", "winType", "winSize" and
                "overlap" parameters.
    """

    def __init__(self, excitationSignal, recordedSignal,
                 # coordinates={'points': [],
                 #            'reference': 'south-west-floor corner',
                 #             'unit': 'm'},
                 method='linear', winType=None, winSize=None, overlap=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._excitation = excitationSignal
        self._recording = recordedSignal
#        self._coordinates = coordinates
        self._methodInfo = {'method': method, 'winType': winType,
                            'winSize': winSize, 'overlap': overlap}
        self._systemSignal = self._calculate_tf_ir(excitationSignal,
                                                   recordedSignal,
                                                   method=method,
                                                   winType=winType,
                                                   winSize=winSize,
                                                   overlap=overlap)
#        self._coord_points_per_channel()
        return

    def _to_dict(self):
        out = {'methodInfo': self.methodInfo,
               'coordinates': self.coordinates}
        return out

    def save(self, dirname=time.ctime(time.time())):
        with zipfile.ZipFile(dirname + '.pytta', 'w') as zdir:
            excit = self.excitation.save('excitation')
            zdir.write(excit)
            os.remove(excit)
            rec = self.recording.save('recording')
            zdir.write(rec)
            os.remove(rec)
            out = self._to_dict()
            out['SignalAddress'] = {'excitation': excit,
                                    'recording': rec}
            with open('ImpulsiveResponse.json', 'w') as f:
                json.dump(out, f, indent=4)
            zdir.write('ImpulsiveResponse.json')
            os.remove('ImpulsiveResponse.json')
        return dirname + '.pytta'

# Properties
    @property
    def excitation(self):
        return self._excitation

    @property
    def inputSignal(self):
        return self._excitation

    @property
    def recording(self):
        return self._recording

    @property
    def outputSignal(self):
        return self._recording

    @property
    def irSignal(self):
        return self._systemSignal

    @property
    def tfSignal(self):
        return self._systemSignal

    @property
    def IR(self):
        return self._systemSignal

    @property
    def TF(self):
        return self._systemSignal

    @property
    def systemSignal(self):
        return self._systemSignal

    @property
    def coordinates(self):
        excoords = []
        for chIndex in range(self.excitation.num_channels()):
            excoords.append(self.excitation.channels[chIndex].coordinates)
        incoords = []
        for chIndex in range(self.inputSignal.num_channels()):
            incoords.append(self.inputSignal.channels[chIndex].coordinates)
        coords = {'excitation': excoords, 'inputSignal': incoords}
        return coords

    @property
    def orientation(self):
        exori = []
        for chIndex in range(self.excitation.num_channels()):
            exori.append(self.excitation.channels[chIndex].orientation)
        inori = []
        for chIndex in range(self.inputSignal.num_channels()):
            inori.append(self.inputSignal.channels[chIndex].orientation)
        oris = {'excitation': exori, 'inputSignal': inori}
        return oris

    @property
    def methodInfo(self):
        return self._methodInfo

# NOW COORDINATES MANAGEMENT DONE DIRECTLY TO THE SIGNALOBJS
# Public methods
#    def set_channels_points(self, channels, points):
#        if isinstance(channels, list):
#            if len(channels) != len(points):
#                raise IndexError("Each value on channels list must have a\
#                                 corresponding [x, y, z] on points list.")
#            else:
#                for idx in range(len(channels)):
#                    self.coordinates['points'][idx] = points[idx]
#        elif isinstance(channels, int):
#            try:
#                self.coordinates['points'][channels-1] = points
#            except IndexError:
#                print('The channel value goes beyond the number of channels,\
#                      the point was appended to the points list.')
#                self.coordinates['points'].append(points)
#        else:
#            raise TypeError("channels parameter must be either an int or\
#                            list of int")
#        return

#    def get_channels_points(self, channels):
#        if isinstance(channels, list):
#            outlist = []
#            for idx in channels:
#                outlist.append(self.coordinates['points'][idx-1])
#            return outlist
#        elif isinstance(channels, int):
#            try:
#                return self.coordinates['points'][channels-1]
#            except IndexError:
#                print('Index out of bounds, returning last channel\'s point')
#                return self.coordinates['points'][-1]
#        else:
#            raise TypeError("channels parameter must be either an int or\
#                            list of int")
#            return

# Private methods
    def _calculate_tf_ir(self, inputSignal, outputSignal, method='linear',
                         winType=None, winSize=None, overlap=None):
        if type(inputSignal) is not type(outputSignal):
            raise TypeError("Only signal-like objects can become an \
                            Impulsive Response.")
        elif inputSignal.samplingRate != outputSignal.samplingRate:
            raise ValueError("Both signal-like objects must have the same\
                             sampling rate.")
        if method == 'linear':
            result = outputSignal / inputSignal

        elif method == 'H1':
            if winType is None:
                winType = 'hann'
            if winSize is None:
                winSize = inputSignal.samplingRate//2
            if overlap is None:
                overlap = 0.5
            result = SignalObj(np.zeros((winSize//2 + 1,
                                         outputSignal.freqSignal.shape[1])),
                               domain='freq',
                               samplingRate=inputSignal.samplingRate)
            if outputSignal.num_channels() > 1:
                if inputSignal.num_channels() > 1:
                    if inputSignal.num_channels()\
                            != outputSignal.num_channels():
                        raise ValueError("Both signal-like objects must have\
                                         the same number of channels.")
                    for channel in range(outputSignal.num_channels()):
                        XY, XX = self._calc_csd_tf(
                                inputSignal.timeSignal[:, channel],
                                outputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        result.freqSignal[:, channel] = XY/XX
                else:
                    for channel in range(outputSignal.num_channels()):
                        XY, XX = self._calc_csd_tf(
                                inputSignal.timeSignal,
                                outputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        result.freqSignal[:, channel] = XY/XX
            else:
                XY, XX = self._calc_csd_tf(
                        inputSignal.timeSignal,
                        outputSignal.timeSignal,
                        inputSignal.samplingRate,
                        winType, winSize, winSize*overlap)
                result.freqSignal = XY/XX

        elif method == 'H2':
            if winType is None:
                winType = 'hann'
            if winSize is None:
                winSize = inputSignal.samplingRate//2
            if overlap is None:
                overlap = 0.5
            result = SignalObj(samplingRate=inputSignal.samplingRate)
            result.domain = 'freq'
            if outputSignal.num_channels() > 1:
                if inputSignal.num_channels() > 1:
                    if inputSignal.num_channels()\
                            != outputSignal.num_channels():
                        raise ValueError("Both signal-like objects must have\
                                         the same number of channels.")
                    for channel in range(outputSignal.num_channels()):
                        YX, YY = self._calc_csd_tf(
                                outputSignal.timeSignal[:, channel],
                                inputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        result.freqSignal[:, channel] = YY/YX
                else:
                    YX, YY = self._calc_csd_tf(
                            outputSignal.timeSignal[:, channel],
                            inputSignal.timeSignal,
                            inputSignal.samplingRate,
                            winType, winSize, winSize*overlap)
                    result.freqSignal[:, channel] = YY/YX
            else:
                YX, YY = self._calc_csd_tf(
                        outputSignal.timeSignal,
                        inputSignal.timeSignal,
                        inputSignal.samplingRate,
                        winType, winSize, winSize*overlap)
                result.freqSignal = YY/YX

        elif method == 'Ht':
            if winType is None:
                winType = 'hann'
            if winSize is None:
                winSize = inputSignal.samplingRate//2
            if overlap is None:
                overlap = 0.5
            result = SignalObj(samplingRate=inputSignal.samplingRate)
            result.domain = 'freq'
            if outputSignal.num_channels() > 1:
                if inputSignal.num_channels() > 1:
                    if inputSignal.num_channels()\
                            != outputSignal.num_channels():
                        raise ValueError("Both signal-like objects must have\
                                         the same number of channels.")
                    for channel in range(outputSignal.num_channels()):
                        XY, XX = self._calc_csd_tf(
                                inputSignal.timeSignal[:, channel],
                                outputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        YX, YY = self._calc_csd_tf(
                                outputSignal.timeSignal[:, channel],
                                inputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        result.freqSignal[:, channel] \
                            = (YY - XX + np.sqrt(
                                    (XX-YY)**2 + 4*np.abs(XY)**2)) / 2*YX

                else:
                    XY, XX = self._calc_csd_tf(
                            inputSignal.timeSignal,
                            outputSignal.timeSignal[:, channel],
                            inputSignal.samplingRate,
                            winType, winSize, winSize*overlap)
                    YX, YY = self._calc_csd_tf(
                            outputSignal.timeSignal[:, channel],
                            inputSignal.timeSignal,
                            inputSignal.samplingRate,
                            winType, winSize, winSize*overlap)
                    result.freqSignal[:, channel]\
                        = (YY - XX + np.sqrt(
                                (XX-YY)**2 + 4*np.abs(XY)**2)) / 2*YX
            else:
                XY, XX = self._calc_csd_tf(
                        inputSignal.timeSignal,
                        outputSignal.timeSignal,
                        inputSignal.samplingRate,
                        winType, winSize, winSize*overlap)
                YX, YY = self._calc_csd_tf(
                        outputSignal.timeSignal,
                        inputSignal.timeSignal,
                        inputSignal.samplingRate,
                        winType, winSize, winSize*overlap)
                result.freqSignal = (YY - XX
                                     + np.sqrt((XX-YY)**2
                                               + 4*np.abs(XY)**2)) / 2*YX

        result.channels = outputSignal.channels / inputSignal.channels
        return result    # end of function get_transferfunction()

    def _calc_csd_tf(self, sig1, sig2, samplingRate, windowName,
                     numberOfSamples, overlapSamples):
        f, S11 = ss.csd(sig1, sig1, samplingRate, window=windowName,
                        nperseg=numberOfSamples, noverlap=overlapSamples,
                        axis=0)
        f, S12 = ss.csd(sig1, sig2, samplingRate, window=windowName,
                        nperseg=numberOfSamples, noverlap=overlapSamples,
                        axis=0)
        return S12, S11

#    def _coord_points_per_channel(self):
#        if len(self.coordinates['points']) == 0:
#            for idx in range(self.IR.num_channels()):
#                self.coordinates['points'].append([0., 0., 0.])
#        elif len(self.coordinates['points']) != self.IR.num_channels():
#            while len(self.coordinates['points']) != self.IR.num_channels():
#                if len(self.coordinates['points']) < self.IR.num_channels():
#                    self.coordinates['points'].append([0., 0., 0.])
#                elif len(self.coordinates['points']) < self.IR.num_channels():
#                    self.coordinates['points'].pop(-1)
#        elif len(self.coordinates['points']) == self.IR.num_channels():
#            pass
#        return


# Measurement class
class Measurement(PyTTaObj):
    """
    Measurement object class created to define some properties and methods to
    be used by the playback, recording and processing classes. It is a private
    class

    Properties(self): (default), (dtype), meaning;

        * device (system default), (list/int):
            list of input and output devices;

        * inChannel ([1]), (ChannelsList | list[int]):
            list of device's input channel used for recording;

        * outChannel ([1]), (ChannelsList | list[int]):
            list of device's output channel used for playing/reproducing\
            a signalObj;

    Properties inherited (default), (dtype): meaning;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * lengthDomain ('time'), (str):
            signal's length domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds for lengthDomain = 'time';

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples for\
            lengthDomain = 'samples';

        * numSamples (samples), (int):
            signal's number of samples

        * freqMin (20), (int):
            minimum frequency bandwidth limit;

        * freqMax (20000), (int):
            maximum frequency bandwidth limit;

        * comment ('No comments.'), (str):
            some commentary about the signal or measurement object;
    """

    def __init__(self,
                 device=None,
                 inChannel=None,
                 outChannel=None,
                 channelName=None,
                 blocking=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # device number. For device list use sounddevice.query_devices()
        self.device = device
        self.inChannel = ChannelsList(inChannel)
        self.outChannel = ChannelsList(outChannel)
        self.blocking = blocking
        return

    def _to_dict(self):
        out = {'device': self.device,
               'inChannel': self.inChannel._to_dict(),
               'outChannel': self.outChannel._to_dict()}
        return out

# Measurement Properties
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, newDevice):
        self._device = newDevice
        return


# RecMeasure class
class RecMeasure(Measurement):
    """
    Recording object

    Properties:
    ------------

        * lengthDomain ('time'), (str):
            signal's length domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds for lengthDomain = 'time';

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples for\
            lengthDomain = 'samples';

        * device (system default), (list/int):
            list of input and output devices;

        * inChannel ([1]), (list/int):
            list of device's input channel used for recording;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * numSamples (samples), (int):
            signal's number of samples

        * freqMin (20), (float):
            minimum frequency bandwidth limit;

        * freqMax (20000), (float):
            maximum frequency bandwidth limit;

        * comment ('No comments.'), (str):
            some commentary about the signal or measurement object;

    Methods:
    ---------

        * run():
            starts recording using the inch and device information, during
            timeLen seconds;
    """

    def __init__(self,
                 lengthDomain=None,
                 fftDegree=None,
                 timeLength=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lengthDomain = lengthDomain
        if self.lengthDomain == 'samples':
            self._fftDegree = fftDegree
        elif self.lengthDomain == 'time':
            self._timeLength = timeLength
        else:
            self._timeLength = None
            self._fftDegree = None
        self._outChannel = None
        return

    def _to_dict(self):
        sup = super()._to_dict()
        sup['fftDegree'] = self.fftDegree
        return sup

    def save(self, dirname=time.ctime(time.time())):
        dic = self._to_dict()
        name = dirname + '.pytta'
        with zipfile.ZipFile(name, 'w') as zdir:
            with open('RecMeasure.json', 'w') as f:
                json.dump(dic, f, indent=4)
            zdir.write('RecMeasure.json')
        return name

# Rec Properties
    @property
    def timeLength(self):
        return self._timeLength

    @timeLength.setter
    def timeLength(self, newLength):
        self._timeLength = np.round(newLength, 2)
        self._numSamples = self.timeLength * self.samplingRate
        self._fftDegree = np.round(np.log2(self.numSamples), 2)
        return

    @property
    def fftDegree(self):
        return self._fftDegree

    @fftDegree.setter
    def fftDegree(self, newDegree):
        self._fftDegree = np.round(newDegree, 2)
        self._numSamples = 2**self.fftDegree
        self._timeLength = np.round(self.numSamples / self.samplingRate, 2)
        return

# Rec Methods
    def run(self):
        """
        Run method: starts recording during Tmax seconds
        Outputs a signalObj with the recording content
        """
        # Record
        recording = sd.rec(self.numSamples,
                           self.samplingRate,
                           mapping=self.inChannel.mapping(),
                           blocking=self.blocking,
                           device=self.device,
                           latency='low',
                           dtype='float32')
        recording = np.squeeze(recording)
        recording = SignalObj(signalArray=recording,
                              domain='time',
                              samplingRate=self.samplingRate)
        recording.channels = self.inChannel
        recording.timeStamp = time.ctime(time.time())
        recording.freqMin, recording.freqMax\
            = (self.freqMin, self.freqMax)
        recording.comment = 'SignalObj from a Rec measurement'
        _print_max_level(recording, kind='input')
        return recording


# PlayRecMeasure class
class PlayRecMeasure(Measurement):
    """
    Playback and Record object

    Properties:
    ------------

        * excitation (SignalObj), (SignalObj):
            signal information used to reproduce (playback);

        * device (system default), (list/int):
            list of input and output devices;

        * inChannel ([1]), (list/int):
            list of device's input channel used for recording;

        * outChannel ([1]), (list/int):
            list of device's output channel used for playing or reproducing
            a signalObj;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * lengthDomain ('time'), (str):
            signal's length domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds for lengthDomain = 'time';

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples for\
            lengthDomain = 'samples';

        * numSamples (samples), (int):
            signal's number of samples

        * freqMin (20), (int):
            minimum frequency bandwidth limit;

        * freqMax (20000), (int):
            maximum frequency bandwidth limit;

        * comment ('No comments.'), (str):
            some commentary about the signal or measurement object;

    Methods: meaning;
        * run():
            starts playing the excitation signal and recording during the
            excitation timeLen duration;
    """

    def __init__(self, excitation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if excitation is None:
            self._excitation = None
        else:
            self.excitation = excitation
            self.outChannel = excitation.channels
        return

# PlayRec Methods
    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Outputs a signalObj with the recording content
        """
        timeStamp = time.ctime(time.time())
        recording = sd.playrec(self.excitation.timeSignal,
                               samplerate=self.samplingRate,
                               input_mapping=self.inChannel.mapping(),
                               output_mapping=self.outChannel.mapping(),
                               device=self.device,
                               blocking=self.blocking,
                               latency='low',
                               dtype='float32')
        recording = np.squeeze(recording)
        recording = SignalObj(signalArray=recording,
                              domain='time',
                              samplingRate=self.samplingRate)
        recording.channels = self.inChannel
        recording.timeStamp = timeStamp
        recording.freqMin, recording.freqMax\
            = (self.freqMin, self.freqMax)
        recording.comment = 'SignalObj from a PlayRec measurement'
        _print_max_level(self.excitation, kind='output')
        _print_max_level(recording, kind='input')
        return recording

    def _to_dict(self):
        sup = super()._to_dict()
        sup['excitationAddress'] = self.excitation._to_dict()
        return sup

    def save(self, dirname=time.ctime(time.time())):
        dic = self._to_dict()
        name = dirname + '.pytta'
        with zipfile.ZipFile(name, 'w') as zdir:
            excit = self.excitation.save('excitation')
            dic['excitationAddress'] = excit
            zdir.write(excit)
            os.remove(excit)
            with open('PlayRecMeasure.json', 'w') as f:
                json.dump(dic, f, indent=4)
            zdir.write('PlayRecMeasure.json')
            os.remove('PlayRecMeasure.json')
        return name

# PlayRec Properties
    @property
    def excitation(self):
        return self._excitation

    @excitation.setter
    def excitation(self, newSignalObj):
        self._excitation = newSignalObj
        return

    @property
    def samplingRate(self):
        return self.excitation._samplingRate

    @property
    def fftDegree(self):
        return self.excitation._fftDegree

    @property
    def timeLength(self):
        return self.excitation._timeLength

    @property
    def numSamples(self):
        return self.excitation._numSamples

    @property
    def freqMin(self):
        return self.excitation._freqMin

    @property
    def freqMax(self):
        return self.excitation._freqMax


# FRFMeasure class
class FRFMeasure(PlayRecMeasure):
    """
    Transferfunction object

    Properties:
    ------------

        * excitation (SignalObj), (SignalObj):
            signal information used to reproduce (playback);

        * device (system default), (list | int):
            list of input and output devices;

        * inChannel ([1]), (list | int):
            list of device's input channel used for recording;

        * outChannel ([1]), (list | int):
            list of device's output channel used for playing or reproducing
            a signalObj;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * lengthDomain ('time'), (str):
            signal's length domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds for lengthDomain = 'time';

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples for\
            lengthDomain = 'samples';

        * numSamples (samples), (int):
            signal's number of samples

        * freqMin (20), (int):
            minimum frequency bandwidth limit;

        * freqMax (20000), (int):
            maximum frequency bandwidth limit;

        * comment ('No comments.'), (str):
            some commentary about the signal or measurement object;

    Methods:
    ---------

        * run():
            starts playing the excitation signal and recording during the
            excitation timeLen duration;
    """

    def __init__(self,
                 coordinates={'points': [],
                              'reference': 'south-west-floor corner',
                              'unit': 'm'},
                 method='linear', winType=None, winSize=None,
                 overlap=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = coordinates
        self.method = method
        self.winType = winType
        self.winSize = winSize
        self.overlap = overlap
        return

    def save(self, dirname=time.ctime(time.time())):
        dic = self._to_dict()
        name = dirname + '.pytta'
        with zipfile.ZipFile(name, 'w') as zdir:
            excit = self.excitation.save('excitation')
            dic['excitationAddress'] = excit
            zdir.write(excit)
            os.remove(excit)
            with open('FRFMeasure.json', 'w') as f:
                json.dump(dic, f, indent=4)
            zdir.write('FRFMeasure.json')
            os.remove('FRFMeasure.json')
        return name

    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Outputs the transferfunction ImpulsiveResponse
        """
        recording = super().run()
        transferfunction = ImpulsiveResponse(self.excitation,
                                             recording,
                                             # self.coordinates,
                                             self.method,
                                             self.winType,
                                             self.winSize,
                                             self.overlap)
        transferfunction.timeStamp = recording.timeStamp
        return transferfunction


# Streaming class
class Streaming(PyTTaObj):
    """
    Wrapper class for SoundDevice stream-like classes. This is intended for
    applications where both measurement and analysis signal must be handled
    at runtime and/or continuously.

    Parameters:
    ------------

        * device:
            Integer or list of integers, the ID number of the desired device to
            reproduce and/or record audio data, as querried by list_devices()
            function.

        * integration:
            The integration period for SPL monitoring, given in seconds.

        * inChannels:
            List of ChannelObj for measurement channels setup

        * outChannels:
            List of ChannelObj for reproduction channels setup. This parameter
            is ignored if `excitation` is provided

        * duration:
            The amount of time that the stream will be active at each start()
            call. This parameter is ignored if `excitation` is provided.

        * excitation:
            A SignalObj used to provide outData, outChannels and samplingRate
            values.

    Attributes:
    ------------

        All parameters are also attributes, along with the ones explained here.

        * inData:
            Recorded audio data (only if `inChannels` provided).

        * outData:
            Audio data used for reproduction (only if `outChannels` provided).

        * active:
            Wrapper for stream.active attribute

        * stopped:
            Wrapper for stream.stopped attribute

        * closed:
            Wrapper for stream.closed attribute

        * stream:
            The actual SoundDevice stream-like object. More information about
            it at http://python-sounddevice.readthedocs.io/

        * durationInSamples:
            Number of recorded samples (only if `duration` provided)

        At least one channels list must be provided for the object
        initialization, either inChannels or outChannels.

    Methods:
    ---------

        * start():
            Wrapper call of stream.start() method

        * stop():
            Wrapper call of stream.stop() method

        * close():
            Wrapper call of stream.close() method

        * get_inData_as_signal():
            Returns the recorded data stored at `inData` as a SignalObj

    Class method:
    ---------------

        * __timeout(obj):
            Class caller for stopping the stream from within callback function

    Callback functions:
    --------------------

        The user can pass his/her own callback function, as long as it have the
        same structure as the ones provided by the Streaming class itself,
        with respect to the number of parameters and its application.

        * __Icallback(Idata, frames, time, status):
            Callback function used for input-only streams:

                * Idata:
                    Numpy array with input audio with `frames` length.

                * frames:
                    Number of frames read at each callback call. Same as
                    `blocksize`.

                * time:
                    Object-like with three timestamps:
                        The first sample read;
                        The last sample read;
                        The callback call.

                * status:
                    PortAudio status flag used to identify if samples were lost
                    due to last callback processing or delayed syscalls

        * __Ocallback(Odata, frames, time, status):
            Callback function used for output-only streams:

                * Odata:
                    An uninitialized Numpy array to be filled with `frames`
                    samples at each call to the callback. This parameter must
                    be full at the callback `return`, if user do not provide
                    enough samples it is filled with zeros. The values must be
                    passed to the parameter in a statement like this:

                        >>> Odata[:] = outputData[:]

                    If no subscription is made on the Odata parameter, the
                    reproduction fails.
            Other parameters are the same as the :method:`__Icallback`

        * __IOcallback(Idata, Odata, frames, time, status):
            Callback function used for input-output streams.
            It\'s parameters are the same as the previous methods.
    """

    def __init__(self,
                 device: List[int] = None,
                 integration: float = None,
                 samplingRate: int = None,
                 inChannels: Optional[List[ChannelObj]] = None,
                 outChannels: Optional[List[ChannelObj]] = None,
                 duration: Optional[float] = None,
                 excitationData: Optional[np.ndarray] = None,
                 callback: Optional[callable] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_channels(inChannels, outChannels, excitationData)
        if duration is not None:
            self._durationInSamples = int(duration*samplingRate)
        else:
            self._durationInSamples = None
        self._inChannels = inChannels
        self._outChannels = outChannels
        self._samplingRate = samplingRate
        self._integration = integration
        self._blockSize = int(self.integration * self.samplingRate)
        self._duration = duration
        self._device = device
        self.__kount = 0
        self.callback = callback
        self._call_for_stream(self.callback)
        return

    def _set_channels(self, inputs, outputs, data):
        if inputs is not None:
            self._inData = np.zeros((1, len(inputs)))
        else:
            self._inData = None
        if outputs is not None:
            try:
                self._outData = data[:]
            except TypeError:
                raise TypeError("If outChannels is provided, an \
                                excitationData must be entered as well.")
        else:
            self._outData = None
        return

    def _call_for_stream(self, IOcallback=None):
        if self.outChannels is not None and self.inChannels is not None:
            if IOcallback is None:
                IOcallback = self.__IOcallback
            self._stream = sd.Stream(self.samplingRate,
                                     self.blockSize,
                                     self.device,
                                     [len(self.inChannels),
                                      len(self.outChannels)],
                                     dtype='float32',
                                     latency='low',
                                     callback=IOcallback)
        elif self.outChannels is not None and self.inChannels is None:
            if IOcallback is None:
                IOcallback = self.__Ocallback
            self._stream = sd.OutputStream(self.samplingRate,
                                           self.blockSize,
                                           self.device,
                                           len(self.outChannels),
                                           dtype='float32',
                                           latency='low',
                                           callback=IOcallback)
        elif self.outChannels is None and self.inChannels is not None:
            if IOcallback is None:
                IOcallback = self.__Icallback
            self._stream = sd.InputStream(self.samplingRate,
                                          self.blockSize,
                                          self.device,
                                          len(self.inChannels),
                                          dtype='float32',
                                          latency='low',
                                          callback=IOcallback)
        else:
            raise ValueError("At least one channel list, either inChannels\
                             or outChannels must be supplied.")
        return

    def __Icallback(self, Idata, frames, time, status):
        self.inData = np.append(self.inData[:]*self.inChannels.CFlist(),
                                Idata, axis=0)
        if self.durationInSamples is None:
            pass
        elif self.inData.shape[0] >= self.durationInSamples:
            self.__timeout()
        return

    def __Ocallback(self, Odata, frames, time, status):
        try:
            Odata[:, :] = self.outData[self.kn:self.kn+frames, :]
            self.kn = self.kn + frames
        except ValueError:
            olen = len(self.outData[self.kn:])
            Odata[:olen, :] = self.outData[self.kn:, :]
            Odata.fill(0)
            self.__timeout()
        return

    def __IOcallback(self, Idata, Odata, frames, time, status):
        self.inData = np.append(self.inData[:]*self.inChannels.CFlist(),
                                Idata, axis=0)
        try:
            Odata[:, :] = self.outData[self.kn:self.kn+frames, :]
            self.kn = self.kn + frames
        except ValueError:
            olen = len(self.outData[self.kn:self.kn+frames])
            Odata[:olen, :] = self.outData[self.kn:, :]
            Odata.fill(0)
            self.__timeout()
        return

    def __timeout(self):
        self.stop()
        self._call_for_stream(self.callback)
        self.kn = 0
        if self.inData is not None:
            self.inData = self.inData[1:, :]
        return

    def getSignal(self):
        signal = SignalObj(self.inData, 'time', self.samplingRate)
        return signal

    def reset(self):
        self.set_channels(self.inChannels, self.outChannels, self.outData)
        return

    def start(self):
        self.stream.start()
        return

    def stop(self):
        self.stream.stop()
        return

    def close(self):
        self.stream.close()
        return

    def calib_pressure(self, chIndex, refSignalObj,
                       refPrms=1.00, refFreq=1000):
        """
        .. method:: calibPressure(chIndex, refSignalObj, refPrms, refFreq):
            use informed SignalObj, with a calibration acoustic pressure
            signal, and the reference RMS acoustic pressure to calculate the
            Correction Factor and apply to every incoming audio on specified
            channel.

            >>> Streaming.calibPressure(chIndex,refSignalObj,refPrms,refFreq)

        Parameters:
        -------------

            * chIndex (), (int):
                channel index for calibration. Starts in 0;

            * refSignalObj (), (SignalObj):
                SignalObj with the calibration recorded signal;

            * refPrms (1.00), (float):
                the reference pressure provided by the acoustic calibrator;

            * refFreq (1000), (int):
                the reference sine frequency provided by the acoustic
                calibrator;
        """

        if chIndex in range(len(self.inChannels)):
            self.inChannels[chIndex].calib_press(
                    refSignalObj, refPrms, refFreq)
            self.inChannels[chIndex].calibCheck = True
        else:
            raise IndexError('chIndex greater than channels number')
        return

    @property
    def stream(self):
        return self._stream

    @property
    def active(self):
        return self.stream.active

    @property
    def stopped(self):
        return self.stream.stopped

    @property
    def closed(self):
        return self.stream.closed

    @property
    def device(self):
        return self._device

    @property
    def inChannels(self):
        return self._inChannels

    @property
    def inData(self):
        return self._inData

    @inData.setter
    def inData(self, data):
        self._inData = data
        return

    @property
    def outChannels(self):
        return self._outChannels

    @property
    def outData(self):
        return self._outData

    @property
    def integration(self):
        return self._integration

    @property
    def blockSize(self):
        return self._blockSize

    @property
    def duration(self):
        return self._durationInSamples/self.samplingRate

    @property
    def durationInSamples(self):
        return self._durationInSamples

    @property
    def kn(self):
        return self.__kount

    @kn.setter
    def kn(self, nk):
        self.__kount = nk
        return


# Sub functions
def _print_max_level(sigObj, kind):
    if kind == 'output':
        for chIndex in range(sigObj.num_channels()):
            print('max output level (excitation) on channel ['
                  + str(chIndex+1) + ']: '
                  + '{:.2f}'.format(sigObj.max_level()[chIndex])
                  + ' ' + sigObj.channels[chIndex].dBName + ' - ref.: '
                  + str(sigObj.channels[chIndex].dBRef)
                  + ' [' + sigObj.channels[chIndex].unit + ']')
            if sigObj.max_level()[chIndex] >= 0:
                print('\x1b[0;30;43mATENTTION! CLIPPING OCCURRED\x1b[0m')
    if kind == 'input':
        for chIndex in range(sigObj.num_channels()):
            print('max input level (recording) on channel ['
                  + str(chIndex+1) + ']: '
                  + '{:.2f}'.format(sigObj.max_level()[chIndex])
                  + ' ' + sigObj.channels[chIndex].dBName
                  + ' - ref.: ' + str(sigObj.channels[chIndex].dBRef)
                  + ' [' + sigObj.channels[chIndex].unit + ']')
            if sigObj.max_level()[chIndex] >= 0:
                print('\x1b[0;30;43mATENTTION! CLIPPING OCCURRED\x1b[0m')
        return


def _to_dict(thing):
    # From SignalObj to dict
    if isinstance(thing, SignalObj):
        mySigObj = vars(thing)
        dictime = {}
        for key, value in mySigObj.items():
            # Recursive stuff for values
            dictime[key] = _to_dict(value)
        # Recursive stuff for resultant dict
        return _to_dict(dictime)

    # From ChannelObj to dict
    elif isinstance(thing, ChannelObj):
        myChObj = vars(thing)
        dictime = {}
        for key, value in myChObj.items():
            dictime[key] = _to_dict(value)
        # Recursive stuff for resultant dict
        return _to_dict(dictime)

    # From a bad dict to a good dict
    elif isinstance(thing, dict):
        dictime = {}
        for key, value in thing.items():
            # Removing spaces from dict keys
            if key.find(' ') >= 0:
                key = key.replace(' ', '')
            # Removing underscores from dict keys
            if key.find('_') >= 0:
                key = key.replace('_', '')
            # Removing empty dicts from values
            if isinstance(value, dict) and len(value) == 0:
                dictime[key] = 0
            # Removing None from values
            if value is None:
                dictime[key] = 0
            # Recursive stuff
            dictime[key] = _to_dict(value)
        return dictime

    # Turning lists into dicts with 'T + listIndex' keys
    elif isinstance(thing, list):
        dictime = {}
        j = 0
        for item in thing:
            dictime['T'+str(j)] = _to_dict(item)
            j = j + 1
        return dictime

    elif thing is None:
        return 0

    else:
        return thing
