# -*- coding: utf-8 -*-
"""
Base classes:
--------------

@Autores:
- Matheus Lazarin Alberto, matheus.lazarin@eac.ufsm.br
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

"""

# Importing modules
from pytta.classes._instanceinfo import RememberInstanceCreationInfo as RICI
import numpy as np
from scipy import io
import time
from pytta import _h5utils as _h5


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
        self._samplingRate = samplingRate
        self._freqMin = freqMin
        self._freqMax = freqMax
        self._comment = comment
        self._lengthDomain = lengthDomain
        self._fftDegree = fftDegree
        self._timeLength = timeLength
        self._numSamples = numSamples

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                # PyTTaObj properties
                f'samplingRate={self.samplingRate!r}, '
                f'freqMin={self.freqMin!r}, '
                f'freqMax={self.freqMax!r}, '
                f'comment={self.comment!r}), '
                f'lengthDomain={self.lengthDomain!r}, '
                f'fftDegree={self.fftDegree!r}, '
                f'timeLength={self.timeLength!r}')

# PyTTaObj Properties
    @property
    def samplingRate(self):
        return int(self._samplingRate)

    @samplingRate.setter
    def samplingRate(self, newSamplingRate):
        self._samplingRate = int(newSamplingRate)
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
    
    # The number of samples depends on the time signal length
    # @numSamples.setter
    # def numSamples(self, newNumSamples):
    #     self._numSamples = newNumSamples
    #     return

    @property
    def freqMin(self):
        return self._freqMin

    @freqMin.setter
    def freqMin(self, newFreqMin):
        self._freqMin = round(newFreqMin/(2**(1/6)), 2)
        return

    @property
    def freqMax(self):
        return self._freqMax

    @freqMax.setter
    def freqMax(self, newFreqMax):
        self._freqMax = round(np.min((newFreqMax*(2**(1/6)),
                                      self.samplingRate//2)), 2)
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
        io.savemat(filename, myObjno_, format='5', oned_as='column')
        return

    def _h5_save(self, h5group):
        h5group.attrs['samplingRate'] = self.samplingRate
        h5group.attrs['freqMin'] = _h5.none_parser(self.freqMin)
        h5group.attrs['freqMax'] = _h5.none_parser(self.freqMax)
        h5group.attrs['comment'] = self.comment
        h5group.attrs['lengthDomain'] = _h5.none_parser(self.lengthDomain)
        h5group.attrs['fftDegree'] = self.fftDegree
        h5group.attrs['timeLength'] = self.timeLength
        pass


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
                # f'polar={self.polar!r}, '
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
        Every parameter becomes the homonym attribute.

        .. attribute:: name:
            String with name or ID;

        .. attribute:: unit:
            String with International System units for the data, e.g. 'Pa', \
            'V', 'FS';

        .. attribute:: CF:
            Calibration factor, numerically convert normalized float32 values \
            to :attr:`unit` values;

        .. attribute:: calibCheck:
            :type:`bool`, information about whether :attr:`CF` is applied \
            (True), or not (False -> default);

    Special methods:
    ------------------

        .. method:: __mul__:
            perform :attr:`unit` concatenation  # TODO unit conversion.

        .. method:: __truediv__:
            perform :attr:`unit` concatenation  # TODO unit conversion.

    """

    units = {'Pa': ('dB', 2e-5),
             'V': ('dBu', 0.775),
             'W/m^2': ('dB', 1e-12),
             'FS': ('dBFS', 1)}

    def __init__(self, num, name=None, code=None, unit='FS', CF=1,
                 calibCheck=False, coordinates=CoordinateObj(),
                 orientation=CoordinateObj()):

        self.num = num
        if name is None:
            self.name = 'Ch. '+str(self.num)
        else:
            self.name = name
        if code is None:
            self.code = self.name[0:2].replace(' ', '')+str(self.num)
        else:
            self.code = code
        self.unit = unit
        self.CF = CF
        self.calibCheck = calibCheck
        self.coordinates = coordinates
        self.orientation = orientation
        return

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num={self.num!r}, '
                f'name={self.name!r}, '
                f'code={self.code!r}, '
                f'unit={self.unit!r}, '
                f'CF={self.CF!r}, '
                f'calibCheck={self.calibCheck!r}, '
                f'coordinates={self.coordinates.point!r}, '
                f'orientation={self.orientation.point!r})')

    def __mul__(self, other):
        if not isinstance(other, ChannelObj):
            raise TypeError('Can\'t "multiply" by other \
                            type than a ChannelObj')
        newCh = ChannelObj(self.num,
                           # name=self.name+'.'+other.name,
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
        Vrms = refSignalObj.rms()[0]
        freqFound = np.round(refSignalObj.freqVector[np.argmax(
                refSignalObj.freqSignal)])
        if not np.isclose(freqFound, float(refFreq), rtol=1e-4):
            print('\x1b[0;30;43mATTENTION! Found calibration frequency (' +
                  '{:.2}'.format(freqFound) +
                  ' [Hz]) differs from refFreq (' +
                  '{:.2}'.format(refFreq) + ' [Hz])\x1b[0m')
        self.CF = refVrms/Vrms
        self.unit = 'V'
        return

    def calib_press(self, refSignalObj, refPrms, refFreq):
        # Prms = np.max(np.abs(refSignalObj.freqSignal[:, 0])) #/ (2**(1/2))
        Prms = refSignalObj.rms()[0]
        freqFound = np.round(refSignalObj.freqVector[np.argmax(
                refSignalObj.freqSignal)])
        if not np.isclose(freqFound, float(refFreq), rtol=1e-4):
            print('\x1b[0;30;43mATTENTION! Found calibration frequency (' +
                  '{}'.format(freqFound) +
                  ' [Hz]) differs from refFreq (' +
                  '{}'.format(refFreq) + ' [Hz])\x1b[0m')
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
            raise ValueError("Channel number must be 1 or greater.")
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
    def code(self):
        return self._code

    @code.setter
    def code(self, newcode):
        if isinstance(newcode, str):
            if ' ' not in newcode:
                self._code = newcode
            else:
                raise TypeError('Channel code cannot contain spaces.')
        else:
            raise TypeError('Channel code must be a string.')
        return

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, newunit):
        if isinstance(newunit, str):
            if newunit in self.units:
                self._unit = newunit
                self.dBName = self.units[newunit][0]
                self.dBRef = self.units[newunit][1]
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
            try:
                self._coordinates.point = newcoord
            except AttributeError:
                self._coordinates = CoordinateObj(point=newcoord)
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
            try:
                self._orientation.point = neworient
            except AttributeError:
                self._orientation = CoordinateObj(point=neworient)
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
    .. class:: ChannelsList(self, chList: list):

        Class to wrap a list of ChannelObj and handle multi-channel SignalObj \
        operations.

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
                    elif (type(memb) is int) or (type(memb) is float):
                        self._channels.append(ChannelObj(int(memb)))
                    else:
                        raise TypeError("Could not resolve ChannelsList initialization parameters.")
            elif type(chList) is int:
                self._channels.append(ChannelObj(chList))
            elif type(chList) is ChannelObj:
                self._channels.append(chList)
            elif type(chList) is ChannelsList:
                self._channels = chList._channels.copy()
                """
                Added .copy() on ChannelsList initialization;
                Implies that:

                    >>> chList1 = pytta.ChannelsList(1)
                    >>> chList2 = pytta.ChannelsList(chList1)

                will make chList2._channels have the same values, and not just
                pointers, to chList1._channels
                """
            else:
                raise TypeError('List initializer must be either positive ' +
                                'int, a list of positive int or ' +
                                'ChannelObj.')
        return

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'chList={self._channels!r})')

    def __len__(self):
        return len(self._channels)

    def __getitem__(self, key):
        if isinstance(key, int):
            try:
                channel = [ch for ch in self._channels if ch.num == key][0]
            except IndexError:
                raise IndexError("Channel number not included.")
        elif isinstance(key, str):
            try:
                channel = [ch for ch in self._channels if ch.name == key or
                           ch.code == key][0]
            except IndexError:
                raise IndexError("Channel name/code out of range.")
        else:
            raise TypeError("Argument must be a channel number (int) or its name (str)")
        return channel


    def __setitem__(self, key, item):
        if isinstance(key, int):
            try:
                channel = [ch for ch in self._channels if ch.num == key][0]
                self._channels.remove(channel)
                self._channels.append(item)
            except IndexError:
                raise IndexError("Channel number not listed.")
        elif isinstance(key, str):
            try:
                channel = [ch for ch in self._channels if ch.name == key or ch.code == key][0]
                self._channels.remove(channel)
                self._channels.append(item)
            except IndexError:
                raise IndexError("Channel name/code not listed.")
        else:
            raise TypeError("Argument must be a channel number (int) or its name (str)")
        return

    def __mul__(self, otherList):
        if not isinstance(otherList, ChannelsList):
            raise TypeError('Can\'t "multiply" by other \
                            type than a ChannelsList')
        if len(self) > 1:
            if len(otherList) > 1:
                if len(self) != len(otherList):
                    raise ValueError("Both ChannelsList-like objects must \
                                     have the same number of channels.")
                newChList = ChannelsList([self[self.mapping[idx]]*
                                          otherList[otherList.mapping[idx]]
                                          for idx in range(len(self))])
            else:
                newChList = ChannelsList([self[self.mapping[idx]]*
                                          otherList[otherList.mapping[0]]
                                          for idx in range(len(self))])
        else:
            if len(otherList) > 1:
                newChList = ChannelsList([self[self.mapping[0]]*otherList[index]
                                          for index in range(len(otherList))])
            else:
                newChList = ChannelsList([self[self.mapping[0]]*
                                          otherList[otherList.mapping[0]]])
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
                
                newChList = ChannelsList([self[self.mapping[idx]]/
                                          otherList[otherList.mapping[idx]]
                                          for idx in range(len(self))])
            else:
                newChList = ChannelsList([self[self.mapping[idx]]/
                                          otherList[otherList.mapping[0]]
                                          for idx in range(len(self))])
        else:
            if len(otherList) > 1:
                newChList = ChannelsList([self[self.mapping[0]]/
                                          otherList[otherList.mapping[idx]]
                                          for idx in range(len(otherList))])
            else:
                newChList = ChannelsList([self._channels[0] /
                                          otherList._channels[0]])
        return newChList

    def __contains__(self, chRef):
        if isinstance(chRef, str):
            if chRef in self.names or chRef in self.codes:
                return True
            else:
                return False
        elif isinstance(chRef, int):
            return chRef in self.mapping

    @property
    def mapping(self):
        return [ch.num for ch in self._channels]

    @property
    def names(self):
        return [ch.name for ch in self._channels]

    @property
    def codes(self):
        return [ch.code for ch in self._channels]

    def CFlist(self):
        out = []
        for obj in self._channels:
            out.append(obj.CF)
        return out

    def dBRefList(self):
        out = []
        for obj in self._channels:
            out.append(obj.dBRef)
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

    def pop(self, Ch=None):
        if Ch not in range(len(self)):
            raise IndexError('Inexistent Channel index')
        elif Ch is None:
            self._channels.pop()
        self._channels.pop(Ch)
        return

    def conform_to(self, rule=None):
        try:
            dCh = rule.numChannels - len(self)
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
