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
from ._instanceinfo import RememberInstanceCreationInfo as RICI
import numpy as np
import scipy.io as sio
from pytta import default, units
import time


class PyTTaObj(RICI):
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
        super().__init__()
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


class CoordinateObj(object):

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
                 coordinates=CoordinateObj(), orientation=CoordinateObj()):
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
        elif isinstance(newcoord, CoordinateObj):
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
        elif isinstance(neworient, CoordinateObj):
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
                newChList = ChannelsList([self[index]*otherList[0]
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
                newChList = ChannelsList([self[index]/otherList[0]
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
        try:
            dCh = rule.num_channels() - len(self)
            # Adjusting number of channels
            if dCh > 0:
                newIndex = len(self)
                for i in range(dCh):
                    self.append(ChannelObj(num=(newIndex+i+1)))
            if dCh < 0:
                for i in range(0, -dCh):
                    self._channels.pop(-1)
        except AttributeError:
            if isinstance(rule, list):
                self._channels = []
                for index in rule:
                    self.append(ChannelObj(num=index+1, name='Channel ' +
                                           str(index)))
            elif rule is None:
                count = 1
                newchs = []
                # Adjusting channel's numbers
                for ch in self._channels:
                    newchs.append(ChannelObj(num=count,
                                             name=ch.name,
                                             unit=ch.unit,
                                             CF=ch.CF,
                                             calibCheck=ch.calibCheck,
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
