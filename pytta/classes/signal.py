# -*- coding: utf-8 -*-

import os
import json
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.io as sio
import sounddevice as sd
import time
from warnings import warn  # , filterwarnings
from pytta import default
from pytta.classes import _base
from pytta import h5utilities as _h5
from pytta.frequtils import fractional_octave_frequencies as FOF

# filterwarnings("default", category=DeprecationWarning)


class SignalObj(_base.PyTTaObj):
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

        * plot_time_dB():
            generates the signal's historic graphic in dB;

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
                 signalArray=np.array([0], ndmin=2, dtype='float32').T,
                 domain='time',
                 *args,
                 **kwargs):
        # Converting signalArray from list to np.array
        if isinstance(signalArray, list):
            signalArray = np.array(signalArray, dtype='float32', ndmin=2).T
        # Checking input array dimensions
        if len(signalArray.shape) > 2:
            message = "No 'pyttaObj' is able handle to arrays with more \
                       than 2 dimensions, '[:,:]', YET!."
            raise AttributeError(message)
        elif len(signalArray.shape) == 1:
            signalArray = np.array(signalArray, ndmin=2, dtype='float32')
        if signalArray.shape[1] > signalArray.shape[0]:
            signalArray = signalArray.T

        super().__init__(*args, **kwargs)

        self.channels = _base.ChannelsList()
        self.lengthDomain = domain
        if self.lengthDomain == 'freq':
            self.freqSignal = signalArray  # [-] signal in frequency domain
        elif self.lengthDomain == 'time':
            self.timeSignal = signalArray  # [-] signal in time domain
        else:
            self.timeSignal = signalArray
            self.lengthDomain = 'time'
            print('Taking the input as a time domain signal.')

        if self.freqMin is None:
            self.freqMin = default.freqMin
        if self.freqMax is None:
            self.freqMax = default.freqMax

        return

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
                newSignal = np.array(newSignal, ndmin=2, dtype='float32')
            if newSignal.shape[1] > newSignal.shape[0]:
                newSignal = newSignal.T
            self._timeSignal = np.array(newSignal, dtype='float32')
            self._freqSignal = np.fft.rfft(self._timeSignal, axis=0, norm=None)
            self._freqSignal = 1/len(self._freqSignal)*self._freqSignal
            # number of samples
            self._numSamples = len(self._timeSignal)
            # size parameter
            self._fftDegree = np.log2(self._numSamples)
            # duration in [s]
            self._timeLength = self.numSamples/self.samplingRate
            # [s] time vector (x axis)
            self._timeVector = np.linspace(0,
                                           self.timeLength 
                                           - 1/self.samplingRate,
                                           self.numSamples)
            # [Hz] frequency vector (x axis)
            self._freqVector = np.linspace(0, (self.numSamples - 1) *
                                           self.samplingRate /
                                           (2*self.numSamples),
                                           (int(self.numSamples/2)+1)
                                           if self.numSamples % 2 == 0
                                           else int((self.numSamples+1)/2))
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
            self._timeSignal = np.array(np.fft.irfft(self._freqSignal,
                                                     axis=0, norm=None),
                                        dtype='float32')
            self._numSamples = len(self.timeSignal)  # [-] number of samples
            self._fftDegree = np.log2(self.numSamples)  # [-] size parameter
            self._timeLength = self.numSamples/self.samplingRate
            # [s] time vector (x axis)
            self._timeVector = np.linspace(0,
                                           self.timeLength 
                                           - 1/self.samplingRate,
                                           self.numSamples)
            self._freqVector = np.linspace(0, (self.numSamples-1) *
                                           self.samplingRate /
                                           (2*self.numSamples),
                                           (int((self.numSamples/2) + 1)
                                           if self.numSamples % 2 == 0
                                           else int((self.numSamples+1)/2)))
            self.channels.conform_to(self)
        else:
            raise TypeError('Input array must be a numpy ndarray')
        return

    @property
    def coordinates(self):
        coords = []
        for chNum in self.channels.mapping:
            coords.append(self.channels[chNum].coordinates)
        return coords

    @property
    def orientation(self):
        orientations = []
        for chNum in self.channels.mapping:
            orientations.append(self.channels[chNum].orientation)
        return orientations

# SignalObj Methods
    def mean(self):
        print('DEPRECATED! This method will be renamed to',
              ':method:``.channelMean()``',
              'Remember to review your code :D')
        return self.channelMean()

    def channelMean(self):
        """
        Returns a signal object with the arithmetic mean channel-wise
        (column-wise) with same number of samples and sampling rate.
        """
        print('New method name in version 0.1.0!',
              'Remember to review your code.')
        return SignalObj(signalArray=np.mean(self.timeSignal, axis=1,
                                             dtype=self.timeSignal.dtype),
                         lengthDomain='time', samplingRate=self.samplingRate)

    @property
    def numChannels(self):
        try:
            numChannels = self.timeSignal.shape[1]
        except IndexError:
            numChannels = 1
        return numChannels

    def num_channels(self):  # DEPRECATED
        warn(DeprecationWarning("This method is DEPRECATED and being " +
                                "replaced by .numChannels property."))
        return self.numChannels

    def max_level(self):
        maxlvl = []
        for chIndex in range(self.numChannels):
            chNum = self.channels.mapping[chIndex]
            maxAmplitude = np.max(self.timeSignal[:, chIndex]**2)
            maxlvl.append(10*np.log10(maxAmplitude /
                                      self.channels[chNum].dBRef**2))
        return maxlvl

    def rms(self):
        return np.mean(self.timeSignal**2, axis=0)**0.5

    def spl(self):
        refList = np.array([ch.dBRef for ch
                            in self.channels], dtype=np.float32)
        return 20*np.log10(self.rms()/refList)

    def size_check(self, inputArray=[]):
        if inputArray.size == 0:
            inputArray = self.timeSignal[:]
        return np.size(inputArray.shape)

    def play(self, outChannels=None, latency='low', **kwargs):
        """
        Play method
        """
        if outChannels is None:
            if self.numChannels <= 1:
                outChannels = default.outChannel
            elif self.numChannels > 1:
                outChannels = np.arange(1, self.numChannels+1)
        sd.play(self.timeSignal, self.samplingRate,
                mapping=outChannels, **kwargs)
        return

    def plot_time(self, xlabel=None, ylabel=None,
                  ylim=None, xlim=None):
        """
        Time domain plotting method
        """
        if xlabel is None:
            xlabel = 'Time [s]'
        if ylabel is None:
            ylabel = 'Amplitude [{}]'
        firstCh = self.channels.mapping[0]
        ylabel = ylabel.format(self.channels[firstCh].unit)
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_axes([0.08, 0.1, 0.8, 0.85], polar=False,
                          projection='rectilinear')
        ax.set_snap(True)
        for chIndex in range(self.numChannels):
            chNum = self.channels.mapping[chIndex]
            label = '{} [{}]'.format(self.channels[chNum].name,
                                     self.channels[chNum].unit)
            ax.plot(self.timeVector,
                    self.timeSignal[:, chIndex],
                    label=label)
        ax.grid(color='gray', linestyle='-.', linewidth=0.4)

        if xlim is None:
            xlim = (self.timeVector[0], self.timeVector[-1])
        ax.set_xlim(xlim)
        
        xticks = np.linspace(*xlim, 11).tolist()
        ax.set_xticks(xticks)
        ax.set_xticklabels(['{:.2n}'.format(tick) for tick in xticks],
                           fontsize=14)
        ax.set_xlabel(xlabel, fontsize=20)

        limData = self.timeSignal
        # limData = [value for value in limData.flatten() if not np.isinf(value)]
        limData[np.isinf(limData)] = 0
        margin = (np.nanmax(limData) - np.nanmin(limData)) / 20
    
        ylimInf = np.nanmin(limData)
        ylimInf -= margin
        
        ylimSup = np.nanmax(limData)
        ylimSup += margin

        if ylim is None:
            ylim = (ylimInf, ylimSup)
        ax.set_ylim(ylim)

        yticks = np.linspace(*ylim, 11).tolist()
        ax.set_yticks(yticks)
        ax.set_yticklabels(['{:n}'.format(float('{0:.2f}'.format(tick)))
                            for tick in yticks], fontsize=14)
        ax.set_ylabel(ylabel, fontsize=20)
        fig.legend(loc='center right', fontsize=12)
        return fig

    def plot_time_dB(self, xlabel=None, ylabel=None,
                     ylim=None, xlim=None):
        """
        Time domain plotting method
        """
        if xlabel is None:
            xlabel = 'Time [s]'
        if ylabel is None:
            ylabel = 'Magnitude [dB]'

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_axes([0.08, 0.15, 0.75, 0.8], polar=False,
                          projection='rectilinear')
        ax.set_snap(True)
        for chIndex in range(self.numChannels):
            chNum = self.channels.mapping[chIndex]
            label = '{} [{}]'.format(self.channels[chNum].name,
                                     self.channels[chNum].unit)
            ax.plot(self.timeVector,
                    10*np.log10(self.timeSignal[:, chIndex]**2 /
                                (self.channels[chNum].dBRef**2)),
                    label=label)
        ax.grid(color='gray', linestyle='-.', linewidth=0.4)

        if xlim is None:
            xlim = (self.timeVector[0], self.timeVector[-1])
        ax.set_xlim(xlim)

        xticks = np.linspace(*xlim, 11).tolist()
        ax.set_xticks(xticks)
        ax.set_xticklabels(['{:.2n}'.format(tick) for tick in xticks],
                           fontsize=14)
        ax.set_xlabel(xlabel, fontsize=20)

        limData = 10*np.log10(self.timeSignal**2/
                              (np.array(self.channels.dBRefList())**2))
        limData = [value for value in limData.flatten() if not np.isinf(value)]
        margin = (np.nanmax(limData) - np.nanmin(limData)) / 20
    
        ylimInf = np.nanmin(limData)
        ylimInf -= margin
        
        ylimSup = np.nanmax(limData)
        ylimSup += margin

        if ylim is None:
            ylim = (ylimInf, ylimSup)
        ax.set_ylim(ylim)

        yticks = np.linspace(*ylim, 11).tolist()
        ax.set_yticks(yticks)
        ax.set_yticklabels(['{:n}'.format(float('{0:.2f}'.format(tick)))
                            for tick in yticks], fontsize=14)
        ax.set_ylabel(ylabel, fontsize=20)
        fig.legend(loc='center right', fontsize=12)
        return fig

    def plot_freq(self, smooth=False, xlabel=None, ylabel=None,
                  ylim=None, xlim=None):
        """
        Frequency domain dB plotting method
        """
        if xlabel is None:
            xlabel = 'Frequency [Hz]'
        if ylabel is None:
            ylabel = 'Magnitude [dB]'

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_axes([0.08, 0.15, 0.75, 0.8], polar=False,
                          projection='rectilinear', xscale='log')
        ax.set_snap(True)
        for chIndex in range(0, self.numChannels):
            chNum = self.channels.mapping[chIndex]
            unitData = '[{} ref.: {} {}]'.format(self.channels[chNum].dBName,
                                                 self.channels[chNum].dBRef,
                                                 self.channels[chNum].unit)
            if smooth:
                Signal = ss.savgol_filter(np.squeeze(np.abs(
                         self.freqSignal[:, chIndex])),
                         31, 3)
            else:
                Signal = self.freqSignal[:, chIndex]  # / (2**(1/2))
            dBSignal = 20 * np.log10(np.abs(Signal) /
                                     self.channels[chNum].dBRef)
            label = '{} {}'.format(self.channels[chNum].name, unitData)
            ax.semilogx(self.freqVector, dBSignal, label=label)
        ax.grid(color='gray', linestyle='-.', linewidth=0.4)

        if xlim is None:
            xlim = (self.freqMin, self.freqMax)

        ax.set_xlim(xlim)
        
        xticks = FOF(minFreq=xlim[0], maxFreq=xlim[1], nthOct=3)[:, 1].tolist()
        ax.set_xticks(xticks)
        ax.set_xticklabels(['{:n}'.format(tick) for tick in xticks],
                           rotation=45, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=20)
        
        freqMinIdx = np.where(self.freqVector > self.freqMin)[0][0]
        freqMaxIdx = np.where(self.freqVector < self.freqMax)[0][-1]

        limData = \
            20*np.log10(np.abs(self.freqSignal[freqMinIdx:freqMaxIdx, :]) /
                        self.channels.dBRefList())

        limData = [value for value in limData.flatten() if not np.isinf(value)]
        margin = (np.nanmax(limData) - np.nanmin(limData)) / 20
    
        ylimInf = np.nanmin(limData)
        ylimInf -= margin
        
        ylimSup = np.nanmax(limData)
        ylimSup += margin
        
        if ylim is None:
            ylim = (ylimInf, ylimSup)
        ax.set_ylim(ylim)

        yticks = np.linspace(*ylim, 11).tolist()
        ax.set_yticks(yticks)
        ax.set_yticklabels(['{:n}'.format(float('{0:.2f}'.format(tick)))
                            for tick in yticks], fontsize=14)
        ax.set_ylabel(ylabel, fontsize=20)
        fig.legend(loc='center right', fontsize=12)
        return fig

    def plot_spectrogram(self, window='hann', winSize=1024, overlap=0.5,
                         xlabel=None, ylabel=None):
        firstCh = self.channels.mapping[0]
        unitData = '[{} ref.: {} {}]'.format(self.channels[firstCh].dBName,
                                             self.channels[firstCh].dBRef,
                                             self.channels[firstCh].unit)

        if xlabel is None:
            xlabel = 'Time in s'
        if ylabel is None:
            ylabel = 'Frequency in Hz'
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_axes([0.1, 0.1, 0.95, 0.8], polar=False,
                          projection='rectilinear')
        ax.set_snap(False)

        _spectrogram, _specTime, _specFreq\
            = self._calc_spectrogram(self.timeSignal[:, 0], overlap,
                                     window, winSize)
        pcmesh = ax.pcolormesh(_specTime, _specFreq, _spectrogram,
                               cmap=plt.jet(), vmin=-120)

        xlim = (self.timeVector[0], self.timeVector[-1])
        ax.set_xlim(xlim)
        xticks = np.linspace(*xlim, 11).tolist()
        ax.set_xticks(xticks)
        ax.set_xticklabels(['{:.2n}'.format(tick) for tick in xticks],
                           fontsize=14)
        ax.set_xlabel(xlabel, fontsize=20)

        ylim = (self.freqMin, self.freqMax)
        ax.set_ylim(ylim)
        yticks = np.linspace(ylim[0], ylim[1], 11).tolist()
        ax.set_yticks(yticks)
        ax.set_yticklabels(['{:n}'.format(float('{0:.2f}'.format(tick)))
                            for tick in yticks], fontsize=14)
        ax.set_ylabel(ylabel, fontsize=20)

        cbar = fig.colorbar(pcmesh)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel('Magnitude {}'.format(unitData),
                           fontsize=20)
        return fig

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
        if chIndex in range(self.numChannels):
            chNum = self.channels.mapping[chIndex]
            self.channels[chNum].calib_volt(refSignalObj, refVrms, refFreq)
            self.timeSignal[:, chIndex] = self.timeSignal[:, chIndex]\
                * self.channels[chNum].CF
            self.channels[chNum].calibCheck = True
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

        if chIndex in range(self.numChannels):
            chNum = self.channels.mapping[chIndex]
            self.channels[chNum].calib_press(refSignalObj, refPrms, refFreq)
            self.timeSignal[:, chIndex] = self.timeSignal[:, chIndex]\
                * self.channels[chNum].CF
            self.channels[chNum].calibCheck = True
        else:
            raise IndexError('chIndex greater than channels number')
        return

    def _to_dict(self):
        out = super()._to_dict()
        out['channels'] = self.channels._to_dict()
        out['timeSignalAddress'] = {'timeSignal': self.timeSignal[:]}
        return out

    def pytta_save(self, dirname=time.ctime(time.time())):
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

    def h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already openned file via
        pytta.save(...).
        """
        h5group.attrs['class'] = 'SignalObj'
        h5group.attrs['channels'] = repr(self.channels)
        h5group['timeSignal'] = self.timeSignal
        super().h5_save(h5group)
        pass

    def __truediv__(self, other):
        """
        Frequency domain division method

        For deconvolution divide by a SignalObj.
        For gain operation divide by a number.

        """
        if type(other) == type(self):
            if other.samplingRate != self.samplingRate:
                raise TypeError("Both SignalObj must have the same sampling rate.")
            result = SignalObj(np.zeros(self.timeSignal.shape),
                            samplingRate=self.samplingRate,
                            freqMin=self.freqMin, freqMax=self.freqMax)
            result.channels = self.channels
            if self.numChannels > 1:
                if other.numChannels > 1:
                    if other.numChannels != self.numChannels:
                        raise ValueError("Both signal-like objects must have the \
                                        same number of channels.")
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(other.numChannels):
                        result_freqSignal[:, channel] = \
                            self.freqSignal[:, channel] \
                            / other.freqSignal[:, channel]
                    result_freqSignal[np.isinf(result_freqSignal)] = 0
                    result.freqSignal = result_freqSignal
                else:
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(self.numChannels):
                        result_freqSignal[:, channel] = \
                            self.freqSignal[:, channel] \
                            / other.freqSignal[:, 0]
                    result_freqSignal[np.isinf(result_freqSignal)] = 0
                    result.freqSignal = result_freqSignal
            else:
                result_freqSignal = self.freqSignal / other.freqSignal
                result_freqSignal[np.isinf(result_freqSignal)] = 0
                result_freqSignal[np.isnan(result_freqSignal)] = 0
                result.freqSignal = result_freqSignal
            result.channels = self.channels / other.channels
        elif type(other) == float or type(other) == int:
            self.timeSignal = self.timeSignal / other
            result = self
        else:
            raise TypeError("A SignalObj can operate with other alike or a " +
                            "number in case of a gain operation.")
        return result

    def __mul__(self, other):
        """
        Gain apply method/FFT convolution
        """
        if type(other) == type(self):
            if other.samplingRate != self.samplingRate:
                raise TypeError("Both SignalObj must have the same sampling rate.")
            result = SignalObj(np.zeros(self.timeSignal.shape),
                            samplingRate=self.samplingRate,
                            freqMin=self.freqMin, freqMax=self.freqMax)
            result.channels = self.channels
            if self.numChannels > 1:
                if other.numChannels > 1:
                    if other.numChannels != self.numChannels:
                        raise ValueError("Both signal-like objects must have the \
                                        same number of channels.")
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(other.numChannels):
                        result_freqSignal[:, channel] = \
                            self.freqSignal[:, channel] \
                            * other.freqSignal[:, channel]
                    result_freqSignal[np.isinf(result_freqSignal)] = 0
                    result.freqSignal = result_freqSignal
                else:
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(self.numChannels):
                        result_freqSignal[:, channel] = \
                            self.freqSignal[:, channel] \
                            * other.freqSignal[:, 0]
                    result_freqSignal[np.isinf(result_freqSignal)] = 0
                    result.freqSignal = result_freqSignal
            else:
                result_freqSignal = self.freqSignal * other.freqSignal
                result_freqSignal[np.isinf(result_freqSignal)] = 0
                result.freqSignal = result_freqSignal
            result.channels = self.channels * other.channels
        elif type(other) == float or type(other) == int:
            self.timeSignal = self.timeSignal * other
            result = self
        else:
            raise TypeError("A SignalObj can operate with other alike or a " +
                            "number in case of a gain operation.")
        return result

    def __add__(self, other):
        """
        Time domain addition method
        """
        result = SignalObj(samplingRate=self.samplingRate)
        result.domain = 'time'
        if isinstance(other, SignalObj):
            if other.samplingRate != self.samplingRate:
                raise TypeError("Both SignalObj must have the same sampling rate.")
            if self.numChannels > 1:
                if other.numChannels > 1:
                    if other.numChannels != self.numChannels:
                        raise ValueError("Both signal-like objects must have\
                                        the same number of channels.")
                    for channel in range(other.numChannels):
                        result.timeSignal = self._timeSignal[:, channel]\
                            + other._timeSignal[:, channel]
                else:
                    for channel in range(other.numChannels):
                        result.timeSignal = self._timeSignal[:, channel]\
                            + other._timeSignal
            else:
                result.timeSignal = self._timeSignal + other._timeSignal
        elif isinstance(other, (float, int)):
            result.timeSignal = self._timeSignal + other
        else:
            raise TypeError("A SignalObj can only operate with other alike, " +
                            "int, or float.")

        result.freqMin, result.freqMax = (self.freqMin, self.freqMax)
        result._channels = self.channels
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
        if self.numChannels > 1:
            if other.numChannels > 1:
                if other.numChannels != self.numChannels:
                    raise ValueError("Both signal-like objects must have\
                                     the same number of channels.")
                for channel in range(other.numChannels):
                    result.timeSignal = self._timeSignal[:, channel]\
                        - other._timeSignal[:, channel]
            else:
                for channel in range(other.numChannels):
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

    def __getitem__(self, key):
        if key > self.numChannels:
            raise IndexError("Index out of bounds.")
        elif key < 0:
            key += self.numChannels
        return SignalObj(self.timeSignal[:, key], 'time', self.samplingRate)

    def _calc_spectrogram(self, timeData=None, overlap=0.5,
                          winType='hann', winSize=1024, *, channel=0):
        if timeData is None:
            timeData = self.timeSignal
            if self.numChannels > 1:
                timeData = timeData[:, channel]
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
class ImpulsiveResponse(_base.PyTTaObj):
    """
        This class is a container of SignalObj, intended to calculate impulsive
        responses and store them.

        The access to this class is provided by itself and as an output
        of the FRFMeasure.run() method.

        Creation parameters:
        ---------------------

            * excitation (SignalObj) (optional)::
                The signal-like object used as excitation signal on the
                measurement-like object. Optional if 'ir' is provided;

            * recording (SignalObj) (optional)::
                The recorded signal-like object, obtained directly from the
                audio interface used on the measurement-like object. Optional
                if 'ir' is provided;

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

            * ir (SignalObj) (optional):
                An calculated impulsive response. Optional if 'excitation' and
                'recording' are provided;

        The class's attribute are described next:

        Attributes:
        ------------

            * irSignal | IR | tfSignal | TF | systemSignal:
                All names are valid, returns the computed impulsive response
                signal-like object;

            * methodInfo:
                Returns a dict with the "method", "winType", "winSize" and
                "overlap" parameters.

        Methods:
        ------------

        * plot_time():
            generates the systemSignal historic graphic;

        * plot_time_dB():
            generates the systemSignal historic graphic in dB;

        * plot_freq():
            generates the systemSignal spectral magnitude graphic;
    """

    def __init__(self, excitation=None, recording=None,
                 method='linear', winType=None, winSize=None, overlap=None,
                 regularization=True, ir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if excitation is None or recording is None:
            if ir is None:
                raise ValueError("You may create an ImpulsiveResponse " +
                                 "passing as parameter the 'excitation' " +
                                 "and 'recording' signals, or a calculated " +
                                 "'ir'.")
            elif not isinstance(ir, SignalObj):
                raise TypeError("'ir' must a SignalObj ")
            self._methodInfo = {'method': method, 'winType': winType,
                                'winSize': winSize, 'overlap': overlap}
            self._systemSignal = ir
        if ir is None:
            if excitation is None or recording is None:
                raise ValueError("You may create an ImpulsiveResponse " +
                                 "passing as parameter the 'excitation' " +
                                 "and 'recording' signals, or a calculated " +
                                 "'ir'.")   
            self._methodInfo = {'method': method, 'winType': winType,
                                'winSize': winSize, 'overlap': overlap}
            self._systemSignal = self._calculate_tf_ir(excitation,
                                                       recording,
                                                       method=method,
                                                       winType=winType,
                                                       winSize=winSize,
                                                       overlap=overlap,
                                                       regularization=
                                                        regularization)
        return

    def __repr__(self):
        method=self.methodInfo['method']
        winType=self.methodInfo['winType']
        winSize=self.methodInfo['winSize']
        overlap=self.methodInfo['overlap']
        return (f'{self.__class__.__name__}('
                f'method={method!r}, '
                f'winType={winType!r}, '
                f'winSize={winSize!r}, '
                f'overlap={overlap!r}, '
                f'ir={self.systemSignal!r}')

    # Methods

    def pytta_save(self, dirname=time.ctime(time.time())):
        with zipfile.ZipFile(dirname + '.pytta', 'w') as zdir:
            ir = self.systemSignal.pytta_save('ir')
            zdir.write(ir)
            os.remove(ir)
            out = self._to_dict()
            out['SignalAddress'] = {'ir': ir}
            with open('ImpulsiveResponse.json', 'w') as f:
                json.dump(out, f, indent=4)
            zdir.write('ImpulsiveResponse.json')
            os.remove('ImpulsiveResponse.json')
        return dirname + '.pytta'

    def h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already openned file via
        pytta.h5_save(...)
        """
        h5group.attrs['class'] = 'ImpulsiveResponse'
        h5group.attrs['method'] = _h5.none_parser(self.methodInfo['method'])
        h5group.attrs['winType'] = _h5.none_parser(self.methodInfo['winType'])
        h5group.attrs['winSize'] = _h5.none_parser(self.methodInfo['winSize'])
        h5group.attrs['overlap'] = _h5.none_parser(self.methodInfo['overlap'])
        self.systemSignal.h5_save(h5group.create_group('systemSignal'))
        pass

    def plot_time(self, *args, **kwargs):
        self.systemSignal.plot_time(*args, **kwargs)

    def plot_time_dB(self, *args, **kwargs):
        self.systemSignal.plot_time_dB(*args, **kwargs)

    def plot_freq(self, *args, **kwargs):
        self.systemSignal.plot_freq(*args, **kwargs)

# Properties

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
    def methodInfo(self):
        return self._methodInfo

    # Private methods

    def _to_dict(self):
        out = {'methodInfo': self.methodInfo}
        return out

    def _calculate_tf_ir(self, inputSignal, outputSignal, method='linear',
                         winType=None, winSize=None, overlap=None,
                         regularization=False):
        if type(inputSignal) is not type(outputSignal):
            raise TypeError("Only signal-like objects can become an \
                            Impulsive Response.")
        elif inputSignal.samplingRate != outputSignal.samplingRate:
            raise ValueError("Both signal-like objects must have the same\
                             sampling rate.")
        if method == 'linear':
            if regularization:
                data = inputSignal.freqSignal
                freqVector = inputSignal.freqVector
                b = data * 0 + 10**(-200/20) # inside signal freq range
                a = data * 0 + 1 # outinside signal freq range
                minFreq = np.max([inputSignal.freqMin, outputSignal.freqMin])
                maxFreq = np.min([inputSignal.freqMax, outputSignal.freqMax])
                # Calculate epsilon
                eps = self._crossfade_spectruns(a, b,
                                               [minFreq/np.sqrt(2),
                                                minFreq],
                                                freqVector)
                if maxFreq < np.min([maxFreq*np.sqrt(2),
                                    inputSignal.samplingRate/2]):
                    eps = self._crossfade_spectruns(eps, a,
                                                    [maxFreq,
                                                    maxFreq*np.sqrt(2)],
                                                    freqVector)
                eps = \
                    eps \
                        * float(np.max(np.abs(outputSignal.freqSignal)))**2 \
                            * 1/2
                C = np.conj(data) / \
                    (np.conj(data)*data + eps)
                C = SignalObj(C,'freq',inputSignal.samplingRate)
                result = outputSignal * C
            else:
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
            if outputSignal.numChannels > 1:
                if inputSignal.numChannels > 1:
                    if inputSignal.numChannels\
                            != outputSignal.numChannels:
                        raise ValueError("Both signal-like objects must have\
                                         the same number of channels.")
                    for channel in range(outputSignal.numChannels):
                        XY, XX = self._calc_csd_tf(
                                inputSignal.timeSignal[:, channel],
                                outputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        result.freqSignal[:, channel] \
                            = np.array(XY/XX, ndmin=2).T
                else:
                    for channel in range(outputSignal.numChannels):
                        XY, XX = self._calc_csd_tf(
                                inputSignal.timeSignal,
                                outputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        result.freqSignal[:, channel] \
                            = np.array(XY/XX, ndmin=2).T
            else:
                XY, XX = self._calc_csd_tf(
                        inputSignal.timeSignal,
                        outputSignal.timeSignal,
                        inputSignal.samplingRate,
                        winType, winSize, winSize*overlap)
                result.freqSignal = np.array(XY/XX, ndmin=2).T

        elif method == 'H2':
            if winType is None:
                winType = 'hann'
            if winSize is None:
                winSize = inputSignal.samplingRate//2
            if overlap is None:
                overlap = 0.5
            result = SignalObj(samplingRate=inputSignal.samplingRate)
            result.domain = 'freq'
            if outputSignal.numChannels > 1:
                if inputSignal.numChannels > 1:
                    if inputSignal.numChannels\
                            != outputSignal.numChannels:
                        raise ValueError("Both signal-like objects must have\
                                         the same number of channels.")
                    for channel in range(outputSignal.numChannels):
                        YX, YY = self._calc_csd_tf(
                                outputSignal.timeSignal[:, channel],
                                inputSignal.timeSignal[:, channel],
                                inputSignal.samplingRate,
                                winType, winSize, winSize*overlap)
                        result.freqSignal[:, channel] \
                            = np.array(YY/YX, ndmin=2).T
                else:
                    YX, YY = self._calc_csd_tf(
                            outputSignal.timeSignal[:, channel],
                            inputSignal.timeSignal,
                            inputSignal.samplingRate,
                            winType, winSize, winSize*overlap)
                    result.freqSignal[:, channel] \
                        = np.array(YY/YX, ndmin=2).T
            else:
                YX, YY = self._calc_csd_tf(
                        outputSignal.timeSignal,
                        inputSignal.timeSignal,
                        inputSignal.samplingRate,
                        winType, winSize, winSize*overlap)
                result.freqSignal = np.array(YY/YX, ndmin=2).T

        elif method == 'Ht':
            if winType is None:
                winType = 'hann'
            if winSize is None:
                winSize = inputSignal.samplingRate//2
            if overlap is None:
                overlap = 0.5
            result = SignalObj(samplingRate=inputSignal.samplingRate)
            result.domain = 'freq'
            if outputSignal.numChannels > 1:
                if inputSignal.numChannels > 1:
                    if inputSignal.numChannels\
                            != outputSignal.numChannels:
                        raise ValueError("Both signal-like objects must have\
                                         the same number of channels.")
                    for channel in range(outputSignal.numChannels):
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
                result.freqSignal = (YY - XX +
                                     np.sqrt((XX-YY)**2 +
                                             4*np.abs(XY)**2)) / 2*YX
        result.freqMin = outputSignal.freqMin
        result.freqMax = outputSignal.freqMax
        result.channels = outputSignal.channels / inputSignal.channels
        return result    # end of function get_transferfunction()

    def _crossfade_spectruns(self, a, b, freqLims, freqVector):
        f0 = freqLims[0]
        f1 = freqLims[1]
        f0idx = np.where(freqVector >= f0)[0][0]
        f1idx = np.where(freqVector <= f1)[0][-1]
        totalSamples = a.shape[0]
        xsamples = f1idx - f0idx
        win = ss.hanning(2*xsamples)

        rightWin = win[xsamples-1:-1]
        fullRightWin = np.concatenate((np.ones(f0idx),
                                       rightWin,
                                       np.zeros(totalSamples-len(rightWin)-f0idx)))

        leftWin = win[0:xsamples]
        fullLeftWin = np.concatenate((np.zeros(f0idx),
                                       leftWin,
                                       np.ones(totalSamples-len(leftWin)-f0idx)))

        aFreqSignal = np.zeros(a.shape, dtype=np.complex_)
        bFreqSignal = np.zeros(b.shape, dtype=np.complex_)
        
        for chIndex in range(a.shape[1]):
            aFreqSignal[:,chIndex] = a[:,chIndex] * fullRightWin
            bFreqSignal[:,chIndex] = b[:,chIndex] * fullLeftWin
        
        a = aFreqSignal
        b = bFreqSignal

        result = a + b

        return result

    def _calc_csd_tf(self, sig1, sig2, samplingRate, windowName,
                     numberOfSamples, overlapSamples):
        f, S11 = ss.csd(sig1, sig1, samplingRate, window=windowName,
                        nperseg=numberOfSamples, noverlap=overlapSamples,
                        axis=0)
        f, S12 = ss.csd(sig1, sig2, samplingRate, window=windowName,
                        nperseg=numberOfSamples, noverlap=overlapSamples,
                        axis=0)
        del f
        return S12, S11
