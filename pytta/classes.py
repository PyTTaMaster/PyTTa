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
from pytta import default
from typing import Optional, List
import time
import copy as cp


class PyTTaObj(object):
    """
    PyTTa object class to define some properties and methods to be used
    by any signal and processing classes. pyttaObj is a private class created
    just to shorten attributes declaration to each PyTTa class.

    Properties(self): (default), (dtype), meaning;

        * samplingRate (44100), (int):
            signal's sampling rate;
        * lengthDomain ('time'), (str):
            input array's domain. May be 'time' or 'samples';
        * timeLength (seconds), (float):
            signal's time length in seconds;
        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples;
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
                 lengthDomain=None,
                 fftDegree=None,
                 timeLength=None,
                 numSamples=None,
                 freqMin=None,
                 freqMax=None,
                 comment="No comments."):

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

    def __dict(self):
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


class ChannelObj(object):
    """
    Base class for signal meta information about the IO channel it's been
    acquired

    Parameters and Attributes:
    ----------------------------
        Every parameter becomes the homonim attribute.

        :param:name :
            String with name or ID;

        :param :`unit`:
            String with International System units for the data, e.g. 'Pa',
            'V', 'FS';

        :param :``CF``:
            Calibration factor, numerically convert normalized float32 values
            to :param:``unit`` values;

        :param:``calibCheck``:
            bool, information about wether CF is applied (True),
            or not (False -> default);

    Special methods:
    ------------------

        :method:``__mul__``:
            perform :param:``unit`` concatenation  # TODO unit conversion.

        :method:``__truediv__``:
            perform :param:``unit`` concatenation  # TODO unit conversion.

    """
    def __init__(self, name='', unit='', CF=1, calibCheck=False):
        self.name = name
        self.unit = unit
        self.CF = CF
        self.calibCheck = calibCheck

    def __mul__(self, other):
        if not isinstance(other, ChannelObj):
            raise TypeError('Can\'t "multiply" by other \
                            type than a ChannelObj')
        newCh = ChannelObj(name=self.name+'.'+other.name,
                           unit=self.unit+'.'+other.unit,
                           CF=self.CF*other.CF,
                           calibCheck=self.calibCheck if self.calibCheck
                           else other.calibCheck)
        return newCh

    def __truediv__(self, other):
        if not isinstance(other, ChannelObj):
            raise TypeError('Can\'t "divide" by other type than a ChannelObj')
        newCh = ChannelObj(name=self.name+'/'+other.name,
                           unit=self.unit+'/'+other.unit,
                           CF=self.CF/other.CF,
                           calibCheck=self.calibCheck if self.calibCheck
                           else other.calibCheck)
        return newCh

# ChannelObj properties

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
            if newunit == 'V':
                self._unit = newunit
                self.dBName = 'dBu'
                self.dBRef = 0.775
            elif newunit == 'Pa':
                self._unit = newunit
                self.dBName = 'dB(z)'
                self.dBRef = 2e-5
            elif newunit == 'W/m^2':
                self._unit = newunit
                self.dBName = 'dB'
                self.dBRef = 1e-12
            elif newunit == '':
                self._unit = ''
                self.dBName = 'dBFs'
                self.dBRef = 1
            else:
                raise TypeError(newunit+' unit not accepted. Must be \'Pa\', \
                                \'V\', \'W/m^2\'  or \'\'.')
        else:
            raise TypeError('Channel unit must be a string.')

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


class ChannelsList(PyTTaObj):

    def __init__(self, chN=0):
        self._channels = []
        if not isinstance(chN, int):
            raise ValueError('Number of channels must be an int')
        for index in range(chN):
            self._channels.append(ChannelObj())
        return

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
        if len(self) != len(otherList):
            raise ValueError('Channels lists must have the same length')
        newChList = ChannelsList()
        for index in range(len(self)):
            newChList.append(self[index]*otherList[index])
        return newChList

    def __truediv__(self, otherList):
        if not isinstance(otherList, ChannelsList):
            raise TypeError('Can\'t "divide" by other \
                            type than a ChannelsList')
        if len(self) != len(otherList):
            raise ValueError('Channels lists must have the same length')
        newChList = ChannelsList()
        for index in range(len(self)):
            newChList.append(self[index]/otherList[index])
        return newChList

    def __dict(self):
        out = {}
        for ch in self._channels:
            out[ch.name] = {'calib': [ch.CF, ch.calibCheck], 'unit': ch.unit}
        return out

    def append(self, newCh):
        if isinstance(newCh, ChannelObj):
            self._channels.append(newCh)
        pass

    def pop(self, Ch):
        if Ch not in range(len(self)):
            raise IndexError('Inexistent Channel index')
        self._channels.pop(Ch)

    def conform_to(self, rule):
        if isinstance(rule, SignalObj):
            dCh = rule.num_channels() - len(self)
            if dCh > 0:
                for i in range(dCh):
                    self.append(ChannelObj(name='Channel '
                                           + str(i+rule.num_channels())))
            if dCh < 0:
                for i in range(0, -dCh):
                    self._channels.pop(-1)
        if isinstance(rule, list):
            self._channels = []
            for index in rule:
                self.append(ChannelObj(name='Channel ' +
                                       str(index)))
        pass

    def _do_rename_channels(self):
        for chIndex in range(len(self)):
            newname = 'Channel ' + str(chIndex+1)
            self._channels[chIndex].name = newname


class SignalObj(PyTTaObj):
    """
    Signal object class.

    Properties:
    ------------

        * domain ('time'), (str):
            domain of the input array;

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

        * samplingRate (44100), (int):
            signal's sampling rate;

        * lengthDomain ('time'), (str):
            input array's domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's duration;

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples;

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
        # Checking input array dimensions
        if self.size_check(signalArray) > 2:
            message = "No 'pyttaObj' is able handle to arrays with more \
                        than 2 dimensions, '[:,:]', YET!."
            raise AttributeError(message)
        elif self.size_check(signalArray) == 1:
            signalArray = np.array(signalArray, ndmin=2).T

        super().__init__(*args, **kwargs)

        self.channels = ChannelsList()

        self.domain = domain or args[1]
        if self.domain == 'freq':
            self.freqSignal = signalArray  # [-] signal in frequency domain
        elif self.domain == 'time':
            self.timeSignal = signalArray  # [-] signal in time domain
        else:
            self.timeSignal = signalArray
            self.domain = 'time'
            print('Taking the input as a time domain signal')

        self.channels.conform_to(self)

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
                newSignal = np.array(newSignal, ndmin=2).T
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
                newSignal = np.array(newSignal, ndmin=2).T
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

# SignalObj Methods
    def mean(self):
        return SignalObj(signalArray=np.mean(self.timeSignal, 1),
                         lengthDomain='time', samplingRate=self.samplingRate)

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

    def __dict(self):
        out = super()._PyTTaObj__dict()
        out['channels'] = self.channels._ChannelsList__dict()
        out['timeSignalAddress'] = {'timeSignal': self.timeSignal[:]}
        return out

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
            self.channels[chIndex].CF = refVrms/Vrms
            self.channels[chIndex].unit = 'V'
            newtimeSignal = cp.deepcopy(self.timeSignal)
            newtimeSignal[:, chIndex] = self.timeSignal[:, chIndex]\
                * self.channels[chIndex].CF
            self.timeSignal = newtimeSignal
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
            self.channels[chIndex].CF = refPrms/Prms
            self.channels[chIndex].unit = 'Pa'
            newtimeSignal = cp.deepcopy(self.timeSignal)
            newtimeSignal[:, chIndex] = self.timeSignal[:, chIndex]\
                * self.channels[chIndex].CF
            self.timeSignal = newtimeSignal
            self.channels[chIndex].calibCheck = True
        else:
            raise IndexError('chIndex greater than channels number')
        return

    def save(self, dirname=time.ctime(time.time())):
        mySigObj = self.__dict()
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
        return

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
                 coordinates={'points': [],
                              'reference': 'south-west-floor corner',
                              'unit': 'm'},
                 method='linear', winType=None, winSize=None, overlap=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._excitation = excitationSignal
        self._recording = recordedSignal
        self._coordinates = coordinates
        self._methodInfo = {'method': method, 'winType': winType,
                            'winSize': winSize, 'overlap': overlap}
        self._systemSignal = self._calculate_tf_ir(excitationSignal,
                                                   recordedSignal,
                                                   method=method,
                                                   winType=winType,
                                                   winSize=winSize,
                                                   overlap=overlap)
        self._coord_points_per_channel()
        return

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
        return self._coordinates

    @property
    def methodInfo(self):
        return self._methodInfo

# Public methods
    def set_channels_points(self, channels, points):
        if isinstance(channels, list):
            if len(channels) != len(points):
                raise IndexError("Each value on channels list must have a\
                                 corresponding [x, y, z] on points list.")
            else:
                for idx in range(len(channels)):
                    self.coordinates['points'][idx] = points[idx]
        elif isinstance(channels, int):
            try:
                self.coordinates['points'][channels-1] = points
            except IndexError:
                print('The channel value goes beyond the number of channels,\
                      the point was appended to the points list.')
                self.coordinates['points'].append(points)
        else:
            raise TypeError("channels parameter must be either an int or\
                            list of int")
        return

    def get_channels_points(self, channels):
        if isinstance(channels, list):
            outlist = []
            for idx in channels:
                outlist.append(self.coordinates['points'][idx-1])
            return outlist
        elif isinstance(channels, int):
            try:
                return self.coordinates['points'][channels-1]
            except IndexError:
                print('Index out of bounds, returning last channel\'s point')
                return self.coordinates['points'][-1]
        else:
            raise TypeError("channels parameter must be either an int or\
                            list of int")
            return

# Private methods
    def _calculate_tf_ir(self, inputSignal, outputSignal, method='linear',
                         winType=None, winSize=None, overlap=None):
        if type(inputSignal) != type(outputSignal):
            raise TypeError("Only signal-like objects can become an\
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
        result.channels = outputSignal.channels[:]
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

    def _coord_points_per_channel(self):
        if len(self.coordinates['points']) == 0:
            for idx in range(self.IR.num_channels()):
                self.coordinates['points'].append([0., 0., 0.])
        elif len(self.coordinates['points']) != self.IR.num_channels():
            while len(self.coordinates['points']) != self.IR.num_channels():
                if len(self.coordinates['points']) < self.IR.num_channels():
                    self.coordinates['points'].append([0., 0., 0.])
                elif len(self.coordinates['points']) < self.IR.num_channels():
                    self.coordinates['points'].pop(-1)
        elif len(self.coordinates['points']) == self.IR.num_channels():
            pass
        return


# Measurement class
class Measurement(PyTTaObj):
    """
    Measurement object class created to define some properties and methods to
    be used by the playback, recording and processing classes. It is a private
    class

    Properties(self): (default), (dtype), meaning;

        * device (system default), (list/int):
            list of input and output devices;

        * inChannel ([1]), (list/int):
            list of device's input channel used for recording;

        * outChannel ([1]), (list/int):
            list of device's output channel used for playing/reproducing\
            a signalObj;

    Properties inherited (default), (dtype): meaning;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * lengthDomain ('time'), (str):
            input array's domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds;

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples;

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
        self.inChannel = inChannel  # input channels
        self.outChannel = outChannel  # output channels
        self.channelName = channelName
        self.blocking = blocking
        return

# Measurement Properties
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, newDevice):
        self._device = newDevice
        return

    @property
    def inChannel(self):
        return self._inChannel

    @inChannel.setter
    def inChannel(self, newInputChannel):
        if not isinstance(newInputChannel, list):
            raise AttributeError('inChannel must be a list; e.g. [1] .')
        else:
            try:
                oldChName = self.channelName
                oldInCh = self.inChannel
                self._inChannel = newInputChannel
                self.channelName = None
                for i in oldInCh:
                    if i in self._inChannel:
                        self.channelName[self._inChannel.index(i)]\
                            = oldChName[oldInCh.index(i)]
            except AttributeError:
                self._inChannel = newInputChannel
        return

    @property
    def outChannel(self):
        return self._outChannel

    @outChannel.setter
    def outChannel(self, newOutputChannel):
        self._outChannel = newOutputChannel

    @property
    def channelName(self):
        return self._channelName

    @channelName.setter
    def channelName(self, channelName):
        if channelName is None:
            self._channelName = []
            for chIndex in range(0, len(self.inChannel)):
                self._channelName.append('Channel ' + str(chIndex + 1))
        elif len(channelName) == len(self.inChannel):
            self._channelName = []
            self._channelName = channelName
        else:
            raise AttributeError('Incompatible number of channel names and\
                                 channel number.')
        return


# RecMeasure class
class RecMeasure(Measurement):
    """
    Recording object

    Properties:
    ------------

        * lengthDomain ('time'), (str):
            input array's domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds;

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples;

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
        return

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
                           mapping=self.inChannel,
                           blocking=self.blocking,
                           device=self.device,
                           latency='low',
                           dtype='float32')
        recording = np.squeeze(recording)
        recording = SignalObj(signalArray=recording,
                              domain='time',
                              samplingRate=self.samplingRate)
        for chIndex in range(self.recording.num_channels()):
            self.recording.channels[chIndex].name = self.channelName[chIndex]
        self.recording.timeStamp = time.ctime(time.time())
        self.recording.freqMin, self.recording.freqMax\
            = (self.freqMin, self.freqMax)
        self.recording.comment = 'SignalObj from a Rec measurement'
        _print_max_level(self.recording, kind='input')
        return self.recording


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
            input array's domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds;

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples;

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
                               input_mapping=self.inChannel,
                               output_mapping=self.outChannel,
                               device=self.device,
                               blocking=self.blocking,
                               latency='low',
                               dtype='float32')
        recording = np.squeeze(recording)
        recording = SignalObj(signalArray=recording,
                              domain='time',
                              samplingRate=self.samplingRate)
        for chIndex in range(self.recording.num_channels()):
            recording.channels[chIndex].name = self.channelName[chIndex]
        recording.timeStamp = timeStamp
        recording.freqMin, self.recording.freqMax\
            = (self.freqMin, self.freqMax)
        recording.comment = 'SignalObj from a PlayRec measurement'
        _print_max_level(self.excitation, kind='output')
        _print_max_level(recording, kind='input')
        return recording

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
            input array's domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds;

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples;

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

    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Outputs the transferfunction ImpulsiveResponse
        """
        recording = super().run()
        transferfunction = ImpulsiveResponse(self.excitation,
                                             recording,
                                             self.coordinates,
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
                 IOcallback: Optional[callable] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_channels(inChannels, outChannels, excitationData)
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
        self.IOcallback = IOcallback
        self._call_for_stream(IOcallback)
        return

    def set_channels(self, inputs, outputs, data):
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
        self.inData = np.append(self.inData, Idata, axis=0)
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
        self.inData = np.append(self.inData[:, :], Idata, axis=0)
        try:
            Odata[:, :] = self.outData[self.kn:self.kn+frames, :]
            self.kn = self.kn + frames
        except ValueError:
            olen = len(self.outData[self.kn:self.kn+frames])
            Odata[:olen, :] = self.outData[self.kn:, :]
            Odata.fill(0)
            self.__timeout()
        return

    def get_inData_as_signal(self):
        signal = SignalObj(self.inData, 'time', self.samplingRate)
        return signal

    def __timeout(self):
        self.stop()
        self._call_for_stream(self.IOcallback)
        self.kn = 0
        if self.inData is not None:
            self.inData = self.inData[1:, :]
        return

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
