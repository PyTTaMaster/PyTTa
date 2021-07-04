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
from pytta import _h5utils as _h5
from pytta.utils import fractional_octave_frequencies as FOF
from pytta import _plot as plot
import copy as cp

# filterwarnings("default", category=DeprecationWarning)
class SignalObj(_base.PyTTaObj):
    """
    Signal object class.
    
    Holds real time signals and their FFT spectra, which are symmetric.
    Therefore only half of the frequency domain signal is stored.

    Creation parameters (default), (type):
    ------------

        * signalArray (ndarray | list), (NumPy array):
            signal at specified domain. For 'freq' domain only half of the
            spectra must be provided. The total numSamples should also
            be provided.

        * domain ('time'), (str):
            domain of the input array. May be 'freq' or 'time'.
            For 'freq' additional inputs should be provided:
                
                * numSamples (len(SignalArray)*2-1), (int):
                    Total signal's number of samples. The default
                    value takes into account a signal with even number of
                    samples.

        * samplingRate (44100), (int):
            signal's sampling rate;

        * signalType ('power'), ('str'):
            type of the input signal. 'power' for finite power signal (infinite
            energy) and 'energy' for energy signal (power tends to zero);

        * freqMin (20), (int):
            minimum frequency bandwidth limit;

        * freqMax (20000), (int):
            maximum frequency bandwidth limit;

        * comment ('No comments.'), (str):
            some commentary about the signal or measurement object;


    Attributes (default), (data type):
    -----------------------------------
        
        * timeSignal (), (NumPy ndarray):
            signal at time domain;

        * timeVector (), (NumPy ndarray):
            time reference vector for timeSignal;

        * freqSignal (), (NumPy ndarray):
            signal at frequency domain;

        * freqVector (), (NumPy ndarray):
            frequency reference vector for freqSignal;
            
        * channels (), (_base.ChannelsList):
            ChannelsList object with info about each SignalObj channel;

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
            signal's number of samples;

        * coordinates (list), (list):
            position in space for the current SignalObj;

        * orientation (list), (list):
            orientation for the current SignalObj;

        * numChannels (int), (int):
            number of channels;


    Methods:
    ---------
    
        * crop(startTime, endTime):
            crops the timeSignal within the provided time interval;

        * max_level():
            return the channel's max levels;

        * rms():
            return the effective value for the entire signal;

        * spl():
            gives the sound pressure level for the entire signal.
            Calibration is needed;

        * play():
            reproduce the timeSignal with default output device;

        * plot_time():
            generates the signal's historic graphic;

        * plot_time_dB():
            generates the signal's historic graphic in dB;

        * plot_freq():
            generates the signal's spectre graphic;

        * plot_spectrogram():
            generates the signal's spectrogram graphic;

        * calib_voltage(refSignalObj,refVrms,refFreq):
            voltage calibration from an input SignalObj;

        * calib_pressure(refSignalObj,refPrms,refFreq):
            pressure calibration from an input SignalObj;

        * save_mat(filename):
            save a SignalObj to a .mat file;
            
    For further information on methods see its specific documentation.
    
    """

    def __init__(self,
                 signalArray=np.array([0], ndmin=2, dtype='float32').T,
                 domain='time',
                 *args,
                 **kwargs):
        # Check if input is a complex array
        if True in np.iscomplex(signalArray):
            dtype = 'complex64'
        else:
            dtype = 'float32'
        # Converting signalArray from list to np.array
        if isinstance(signalArray, list):
            signalArray = np.array(signalArray, dtype=dtype, ndmin=2).T
        # Checking input array dimensions
        if len(signalArray.shape) > 2:
            message = "No 'pyttaObj' is able handle to arrays with more \
                       than 2 dimensions, '[:,:]', YET!."
            raise AttributeError(message)
        elif len(signalArray.shape) == 1:
            signalArray = np.array(signalArray, ndmin=2, dtype=dtype)
        if signalArray.shape[1] > signalArray.shape[0]:
            signalArray = signalArray.T

        if 'signalType' in kwargs:
            signalType = kwargs.pop('signalType')
        else:
            signalType = 'power'
            
        if domain == 'freq':
            if 'numSamples' in kwargs:
                self._numSamples = kwargs.pop('numSamples')
            else:
                # Consider the full signal has EVEN numSamples
                halfSpectraNumSamples = len(signalArray)
                # self._numSamples = halfSpectraNumSamples*2-1  # ODD
                self._numSamples = (halfSpectraNumSamples-1)*2  # EVEN

        super().__init__(*args, **kwargs)

        self.channels = _base.ChannelsList()
        self.signalType = signalType
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
    def signalType(self):
        return self._signalType

    @signalType.setter
    def signalType(self, newSigType):
        if not isinstance(newSigType, str):
            raise TypeError("signalType must be a string and may be 'power' " +
                            "for music, speech, and other recorded signals, " +
                            "or 'energy' for impulsive responses.")
        elif newSigType not in ['power', 'energy']:
            raise TypeError("signalType may be 'power' " +
                            "for music, speech, and other recorded signals, " +
                            "or 'energy' for impulsive responses.")
        elif hasattr(self, '_signalType'):
            if newSigType == self.signalType:
                print("'signalType' is already '" + self.signalType + "'.")
            else:
                self._signalType = newSigType
                self._fft()
        # for initialization purposes
        else:
            self._signalType = newSigType

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
            self._numSamples = len(self._timeSignal)  # [-] number of samples
            self._fftDegree = np.log2(self._numSamples)  # [-] size parameter
            self._timeLength = self.numSamples/self.samplingRate  # [s]
            self._timeVector = np.linspace(0,
                                           self.timeLength
                                           - 1/self.samplingRate,
                                           self.numSamples)
            self._fft()
            self.channels.conform_to(self)
        else:
            raise TypeError('Input array must be a numpy ndarray')
        return

    @property
    def freqSignal(self):
        """
        Return half of the RMS spectrum. Normalized in case of a power signal.
        """
        return self._freqSignal

    @freqSignal.setter
    def freqSignal(self, newSignal):
        if isinstance(newSignal, np.ndarray):
            if self.size_check(newSignal) == 1:
                newSignal = np.array(newSignal, ndmin=2)
            if newSignal.shape[1] > newSignal.shape[0]:
                newSignal = newSignal.T
            # Time numSamples was provided (or is default) at init or old
            # signal was already here. Check if numSamples calculated
            # according to half spectra matches the current numSamples.
            halfSpectraNumSamples = len(newSignal)
            if (halfSpectraNumSamples-1)*2 == self._numSamples:
                timeSignalNumSamplesIs = "EVEN"
            elif halfSpectraNumSamples*2-1 == self._numSamples:
                timeSignalNumSamplesIs = "ODD"
            else:
                # Old numSamples don't match with provided half spectrum
                # number of samples.
                timeSignalNumSamplesIs = "UNKNOWN"  
                
            if timeSignalNumSamplesIs == "UNKNOWN":
                # Consider full spectrum has even number of samples
                # self._numSamples = halfSpectraNumSamples*2-1  # ODD
                self._numSamples = (halfSpectraNumSamples-1)*2  # EVEN
            
            self._freqSignal = np.array(newSignal, dtype='complex64')                                   
            self._fftDegree = np.log2(self.numSamples)  # [-] size parameter
            self._timeLength = self.numSamples/self.samplingRate
            self._freqVector = np.fft.rfftfreq(n=self.numSamples,
                                               d=1/self.samplingRate)
            self._ifft()
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

    # SignalObj Methods
    
    def split(self, channels: list = None) -> list:
        """
        Split the SignalObj channels into several SignalObjs. If the 'channels'
        input argument is given split the specified channel numbers, otherwise
        split all channels.
        
        Arguments (default), (type):
        -----------------------------
        
            * channels (None), (list):
                specified channels to split from the SignalObj;
                
        Return (type):
        --------------
        
            * spltdChs (list):
                a list containing SignalObjs for each specified/all channels;        
            
        """
        
        if channels is None:
            channels = self.channels.mapping
            
        else:
            treta = False
            inexistentChs = []
            for chNum in channels:
                if chNum not in self.channels.mapping:
                    treta = True
                    inexistentChs.append(chNum)
            if treta:
                raise IndexError("Channel number(s) " + str(inexistentChs) +
                                 " don't exist.")
        
        indexes = [self.channels.mapping.index(chNum) for chNum in channels]
            
        spltdChs = []
        
        idx = 0;
        for chNum in channels:
            newSignal = SignalObj(self.timeSignal[:,indexes[idx]],
                                  domain='time',
                                  samplingRate=self.samplingRate,
                                  freqMin=self.freqMin,
                                  freqMax=self.freqMax,
                                  comment=self.comment)
            newSignal.channels[1] = self.channels[chNum]
            
            spltdChs.append(newSignal)
            
            idx += 1
                    
        return spltdChs

    def crop(self, startTime, endTime):
        """crop crop the signal duration in the specified interval

        :param startTime: start time for cropping
        :type startTime: int, float
        :param endTime: end time for cropping
        :type endTime: int, float or str
        """
        if not isinstance(startTime, (float, int)) or \
            not isinstance(endTime, (float, int, str)):
            raise TypeError("'startTime' and 'endTime' must be int, float or " +
                            "'end'.")
        if isinstance(endTime, str):
            if endTime == 'end':
                endTime = self.timeVector[-1]
            else:
                raise TypeError("'endTime' must be int, float or " +
                                "'end'.")
        endIdx = np.where(self.timeVector >= endTime)[0][0]
        startIdx = np.where(self.timeVector >= startTime)[0][0]
        self.timeSignal = self.timeSignal[startIdx:endIdx,:]

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
        return 20*np.log10(self.rms()/self.channels.dBRefList())

    def size_check(self, inputArray=[]):
        if inputArray.size == 0:
            inputArray = self.timeSignal[:]
        return np.size(inputArray.shape)

    def play(self,
             channels: list = None, 
             mapping: list = None,
             latency='low',
             **kwargs):
        """
        Play method.
        
        Only one SignalObj channel can be played through each sound card
        output channel. Check the input parameters below
        
        For usage insights, check the examples folder.
        
        Parameters (default), (type):
        ------------------------------
        
            * channels (None), (list):
                list of channel numbers to play. If not specified all existent
                channels will be chosen;
        
            * mapping (None), (list):
                list of channel numbers of your sound card (starting with 1)
                where the specified channels of the SignalObj shall be played
                back on. Must have the same length as number of SignalObj's
                specified channels (except if SignalObj is mono, in which case
                the signal is played back on all possible output channels).
                Each channel number may only appear once in mapping;
                
        
        """
        if channels is None:
            channels = self.channels.mapping
            
        else:
            treta = False
            inexistentChs = []
            for chNum in channels:
                if chNum not in self.channels.mapping:
                    treta = True
                    inexistentChs.append(chNum)
            if treta:
                raise IndexError("SignalObj channel number(s) " +
                                 str(inexistentChs) +
                                 " don't exist.")
        
        indexes = [self.channels.mapping.index(chNum) for chNum in channels]
        
        timeSignalSel = self.timeSignal[:,indexes[0]]
        
        if len(channels) > 1: 
            for idx in indexes[1:]:
                timeSignalSel = np.vstack((timeSignalSel,
                                           self.timeSignal[:,idx]))
        
        timeSignalSel = timeSignalSel.T
        
        if mapping is None:
            if self.numChannels <= 1:
                mapping = default.outChannel
            elif self.numChannels > 1:
                mapping = np.arange(1, self.numChannels+1)
                
        sd.play(timeSignalSel, self.samplingRate,
                mapping=mapping, **kwargs)
        
        return

    def plot_time(self, xLabel:str=None, yLabel:str=None,
                yLim:list=None, xLim:list=None, title:str=None,
                decimalSep:str=',',timeUnit:str='s'):
        """Plots the signal in time domain.

        xLabel, yLabel, and title are saved for the next plots when provided.

        Parameters (default), (type):
        ------------------------------

            * xLabel ('Time [s]'), (str):
                x axis label.

            * yLabel ('Amplitude'), (str):
                y axis label.

            * yLim (), (list):
                inferior and superior limits.

                >>> yLim = [-100, 100]

            * xLim (), (list):
                left and right limits

                >>> xLim = [0, 15]

            * title (), (str):
                plot title

            * decimalSep (','), (str):
                may be dot or comma.

                >>> decimalSep = ',' # in Brazil

             * timeUnit ('s'), (str):
                'ms' or 's'.


        Return:
        --------

            matplotlib.figure.Figure object.
        """
        if xLabel is not None:
            self.timeXLabel = xLabel
        else:
            if hasattr(self, 'timeXLabel'):
                if self.timeXLabel is not None:
                    xLabel = self.timeXLabel

        if yLabel is not None:
            self.timeYLabel = yLabel
        else:
            if hasattr(self, 'timeYLabel'):
                if self.timeYLabel is not None:
                    yLabel = self.timeYLabel

        if title is not None:
            self.timeTitle = title
        else:
            if hasattr(self, 'timeTitle'):
                if self.timeTitle is not None:
                    title = self.timeTitle

        fig = plot.time((self,), xLabel, yLabel, yLim, xLim, title, decimalSep,
                        timeUnit)
        return fig

    def plot_time_dB(self, xLabel:str=None, yLabel:str=None,
                yLim:list=None, xLim:list=None, title:str=None,
                decimalSep:str=',', timeUnit:str='s'):
        """Plots the signal in decibels in time domain.

        xLabel, yLabel, and title are saved for the next plots when provided.

        Parameters (default), (type):
        ------------------------------

            * xLabel ('Time [s]'), (str):
                x axis label.

            * yLabel ('Amplitude'), (str):
                y axis label.

            * yLim (), (list):
                inferior and superior limits.

                >>> yLim = [-100, 100]

            * xLim (), (list):
                left and right limits

                >>> xLim = [0, 15]

            * title (), (str):
                plot title

            * decimalSep (','), (str):
                may be dot or comma.

                >>> decimalSep = ',' # in Brazil

            * timeUnit ('s'), (str):
            'ms' or 's'.


        Return:
        --------

            matplotlib.figure.Figure object.
        """
        if xLabel is not None:
            self.timedBXLabel = xLabel
        else:
            if hasattr(self, 'timedBXLabel'):
                if self.timedBXLabel is not None:
                    xLabel = self.timedBXLabel

        if yLabel is not None:
            self.timedBYLabel = yLabel
        else:
            if hasattr(self, 'timedBYLabel'):
                if self.timedBYLabel is not None:
                    yLabel = self.timedBYLabel

        if title is not None:
            self.timedBTitle = title
        else:
            if hasattr(self, 'timedBTitle'):
                if self.timedBTitle is not None:
                    title = self.timedBTitle

        fig = plot.time_dB((self,), xLabel, yLabel, yLim, xLim, title,
                           decimalSep, timeUnit)
        return fig

    def plot_freq(self, smooth:bool=False, xLabel:str=None, yLabel:str=None,
                  yLim:list=None, xLim:list=None, title:str=None,
                  decimalSep:str=','):
        """Plots the signal decibel magnitude in frequency domain.

        xLabel, yLabel, and title are saved for the next plots when provided.

        Parameters (default), (type):
        -----------------------------

            * smooth (False), (bool):
                option for curve smoothing. Uses scipy.signal.savgol_filter.
                Preliminar implementation. Needs review.

            * xLabel ('Time [s]'), (str):
                x axis label.

            * yLabel ('Amplitude'), (str):
                y axis label.

            * yLim (), (list):
                inferior and superior limits.

                >>> yLim = [-100, 100]

            * xLim (), (list):
                left and right limits

                >>> xLim = [15, 21000]

            * title (), (str):
                plot title

            * decimalSep (','), (str):
                may be dot or comma.

                >>> decimalSep = ',' # in Brazil

        Return:
        --------

            matplotlib.figure.Figure object.
        """
        if xLabel is not None:
            self.freqXLabel = xLabel
        else:
            if hasattr(self, 'freqXLabel'):
                if self.freqXLabel is not None:
                    xLabel = self.freqXLabel

        if yLabel is not None:
            self.freqYLabel = yLabel
        else:
            if hasattr(self, 'freqYLabel'):
                if self.freqYLabel is not None:
                    yLabel = self.freqYLabel

        if title is not None:
            self.freqTitle = title
        else:
            if hasattr(self, 'freqTitle'):
                if self.freqTitle is not None:
                    title = self.freqTitle

        fig = plot.freq((self,), smooth, xLabel, yLabel, yLim, xLim, title,
                        decimalSep)
        return fig

    def plot_spectrogram(self, winType:str='hann', winSize:int=1024,
                         overlap:float=0.5, xLabel:str=None, yLabel:str=None,
                         yLim:list=None, xLim:list=None, title:str=None,
                         decimalSep:str=','):
        """Plots a signal spectrogram.

        xLabel, yLabel, and title are saved for the next plots when provided.

        Parameters (default), (type):
        -----------------------------

            * winType ('hann'), (str):
                window type for the time slicing.

            * winSize (1024), (int):
                window size in samples

            * overlap (0.5), (float):
                window overlap in %

            * xLabel ('Time [s]'), (str):
                x axis label.

            * yLabel ('Frequency [Hz]'), (str):
                y axis label.

            * yLim (), (list):
                inferior and superior frequency limits.

                >>> yLim = [20, 1000]

            * xLim (), (list):
                left and right time limits

                >>> xLim = [1, 3]

            * title (), (str):
                plot title

            * decimalSep (','), (str):
                may be dot or comma.

                >>> decimalSep = ',' # in Brazil

        Return:
        --------

            List of matplotlib.figure.Figure objects for each item in curveData.
        """
        if xLabel is not None:
            self.spectrogramXLabel = xLabel
        else:
            if hasattr(self, 'spectrogramXLabel'):
                if self.spectrogramXLabel is not None:
                    xLabel = self.spectrogramXLabel

        if yLabel is not None:
            self.spectrogramYLabel = yLabel
        else:
            if hasattr(self, 'spectrogramYLabel'):
                if self.spectrogramYLabel is not None:
                    yLabel = self.spectrogramYLabel

        if title is not None:
            self.spectrogramTitle = title
        else:
            if hasattr(self, 'spectrogramTitle'):
                if self.spectrogramTitle is not None:
                    title = self.spectrogramTitle

        figs = plot.spectrogram((self,), winType, winSize,
                                overlap, xLabel, yLabel, xLim, yLim,
                                title, decimalSep)
        return figs

    def calib_voltage(self, chIndex, refSignalObj, refVrms=1, refFreq=1000):
        """
        Use informed SignalObj with a calibration voltage  signal, and the
        reference RMS voltage to calculate and apply the Correction Factor.

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
            self._fft()
        else:
            raise IndexError('chIndex greater than channels number')
        return

    def calib_pressure(self, chIndex, refSignalObj,
                       refPrms=1.00, refFreq=1000):
        """
        Use informed SignalObj, with a calibration
        acoustic pressure signal, and the reference RMS acoustic pressure to
        calculate and apply the Correction Factor.

            >>> SignalObj.calibPressure(chIndex,refSignalObj,refPrms,refFreq)

        Parameters (default), (type):
        ------------------------------

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
            self._fft()
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

    def _h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already opened file via
        pytta.save(...).
        """
        h5group.attrs['class'] = 'SignalObj'
        h5group.attrs['channels'] = repr(self.channels)
        h5group.attrs['signalType'] = _h5.attr_parser(self.signalType)
        h5group['timeSignal'] = self.timeSignal
        super()._h5_save(h5group)
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
            currentFreqSignal = _make_pk_spectra(self.freqSignal)
            otherFreqSignal = _make_pk_spectra(other.freqSignal)
            result = SignalObj(np.zeros(self.timeSignal.shape),
                               samplingRate=cp.copy(self.samplingRate),
                               freqMin=cp.copy(self.freqMin),
                               freqMax=cp.copy(self.freqMax),
                               signalType='energy')
            result.channels = cp.deepcopy(self.channels)
            if self.numChannels > 1:
                if other.numChannels > 1:
                    if other.numChannels != self.numChannels:
                        raise ValueError("Both signal-like objects must have the \
                                        same number of channels.")
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(other.numChannels):
                        result_freqSignal[:, channel] = \
                            currentFreqSignal[:, channel] \
                            / otherFreqSignal[:, channel]
                else:
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(self.numChannels):
                        result_freqSignal[:, channel] = \
                            currentFreqSignal[:, channel] \
                            / otherFreqSignal[:, 0]
            else:
                result_freqSignal = currentFreqSignal / otherFreqSignal
            result_freqSignal[np.isinf(result_freqSignal)] = 0
            result_freqSignal[np.isnan(result_freqSignal)] = 0
            result.freqSignal = _make_rms_spectra(result_freqSignal)
            result.channels = self.channels / other.channels
        elif type(other) == float or type(other) == int:
            result = SignalObj(np.zeros(self.timeSignal.shape),
                               samplingRate=cp.copy(self.samplingRate),
                               freqMin=cp.copy(self.freqMin),
                               freqMax=cp.copy(self.freqMax),
                               signalType='energy')
            result.channels = cp.deepcopy(self.channels)
            result.timeSignal = self.timeSignal / other

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
            if other.signalType == 'energy' or self.signalType == 'energy':
                signalType = 'energy'
            else:
                signalType = 'power'
            currentFreqSignal = _make_pk_spectra(self.freqSignal)
            otherFreqSignal = _make_pk_spectra(other.freqSignal)
            result = SignalObj(np.zeros(self.timeSignal.shape),
                            samplingRate=cp.copy(self.samplingRate),
                            freqMin=cp.copy(self.freqMin),
                            freqMax=cp.copy(self.freqMax),
                            signalType=signalType)
            result.channels = cp.deepcopy(self.channels)
            if self.numChannels > 1:
                if other.numChannels > 1:
                    if other.numChannels != self.numChannels:
                        raise ValueError("Both signal-like objects must have the \
                                        same number of channels.")
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(other.numChannels):
                        result_freqSignal[:, channel] = \
                            currentFreqSignal[:, channel] \
                            * otherFreqSignal[:, channel]
                else:
                    result_freqSignal = np.zeros(self.freqSignal.shape,
                                                dtype=np.complex_)
                    for channel in range(self.numChannels):
                        result_freqSignal[:, channel] = \
                            currentFreqSignal[:, channel] \
                            * otherFreqSignal[:, 0]
            else:
                result_freqSignal = currentFreqSignal * otherFreqSignal
            result_freqSignal[np.isinf(result_freqSignal)] = 0
            result.freqSignal = _make_rms_spectra(result_freqSignal)
            result.channels = self.channels * other.channels
        elif type(other) == float or type(other) == int:
            result = SignalObj(np.zeros(self.timeSignal.shape),
                            samplingRate=cp.copy(self.samplingRate),
                            freqMin=cp.copy(self.freqMin),
                            freqMax=cp.copy(self.freqMax),
                            signalType=cp.copy(self.signalType))
            result.timeSignal = self.timeSignal * other
            result.channels = cp.deepcopy(self.channels)
        else:
            raise TypeError("A SignalObj can operate with other alike or a " +
                            "number in case of a gain operation.")
        return result

    def __add__(self, other):
        """
        Time domain addition method
        """
        if other.signalType == 'energy' or self.signalType == 'energy':
            signalType = 'energy'
        else:
            signalType = 'power'
        result = SignalObj(np.zeros(self.timeSignal.shape),
                        samplingRate=cp.copy(self.samplingRate),
                        freqMin=cp.copy(self.freqMin),
                        freqMax=cp.copy(self.freqMax),
                        signalType=signalType)
        result.channels = cp.deepcopy(self.channels)
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
        return result

    def __sub__(self, other):
        """
        Time domain subtraction method
        """
        if type(other) != type(self):
            raise TypeError("A SignalObj can only operate with other alike.")
        if other.samplingRate != self.samplingRate:
            raise TypeError("Both SignalObj must have the same sampling rate.")
        if other.signalType == 'energy' or self.signalType == 'energy':
            signalType = 'energy'
        else:
            signalType = 'power'
        result = SignalObj(np.zeros(self.timeSignal.shape),
                        samplingRate=cp.copy(self.samplingRate),
                        freqMin=cp.copy(self.freqMin),
                        freqMax=cp.copy(self.freqMax),
                        signalType=signalType)
        result.channels = cp.deepcopy(self.channels)
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
        ret = SignalObj(self.timeSignal[:, key], 'time', self.samplingRate)
        chnum = self.channels.mapping[key]
        ret.channels[1] = self.channels[chnum]
        return ret

    def _fft(self):
        """fft do the transformation to the frequency domain of the current
        time signal.
        """
        # FFT
        newFreqSignal = \
            np.fft.rfft(self._timeSignal, axis=0, norm=None)
        # turning peak amplitude into RMS amplitude
        newFreqSignal = _make_rms_spectra(newFreqSignal)
        # spectrum normalization
        if self.signalType == 'power':
            newFreqSignal /= len(newFreqSignal)
        # assign new freq signal
        self._freqSignal = newFreqSignal
        # frequency vector (x axis)
        self._freqVector = np.fft.rfftfreq(n=self.numSamples,
                                           d=1/self.samplingRate)
        return

    def _ifft(self):
        """ifft do the transformation to the time domain of the current
        frequency signal
        """
        # spectrum denormalization
        if self.signalType == 'power':
            adjustedFreqSignal = \
                self._freqSignal*len(self._freqSignal)
        else:
            adjustedFreqSignal = self._freqSignal
        # turning RMS amplitude into peak amplitude except DC freq
        adjustedFreqSignal = _make_pk_spectra(adjustedFreqSignal)
        # IFFT
        self._timeSignal = \
            np.array(np.fft.irfft(adjustedFreqSignal,
                                  n=self.numSamples, axis=0, norm=None),
                    dtype='float32')
        # time vector (x axis)
        self._timeVector = np.linspace(0,
                                       self.timeLength
                                       - 1/self.samplingRate,
                                       self.numSamples)
        return


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

        * regularization (bool), (True):
            Do Kirkeby regularization with a packing filter for the impulsive
            response's time signature. Details in 'Advancements in impulsive
            response measurements by sine sweeps' Farina, 2007.

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
            # Zero padding
            elif excitation.numSamples > recording.numSamples:
                print("Zero padding on IR calculation!")
                excitSig = excitation.timeSignal
                recSig = recording.timeSignal
                newTimeSignal = \
                    np.zeros((excitSig.shape[0], recSig.shape[1]))
                newTimeSignal[:recSig.shape[0], :recSig.shape[1]] = \
                    recSig
                recording.timeSignal = newTimeSignal
            elif excitation.numSamples < recording.numSamples:
                print("Zero padding on IR calculation!")
                excitSig = excitation.timeSignal
                recSig = recording.timeSignal
                newTimeSignal = \
                    np.zeros((recSig.shape[0], excitSig.shape[1]))
                newTimeSignal[:excitSig.shape[0], :excitSig.shape[1]] = \
                    excitSig
                excitation.timeSignal = newTimeSignal
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

    def _h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already opened file via
        pytta._h5_save(...)
        """
        h5group.attrs['class'] = 'ImpulsiveResponse'
        h5group.attrs['method'] = _h5.none_parser(self.methodInfo['method'])
        h5group.attrs['winType'] = _h5.none_parser(self.methodInfo['winType'])
        h5group.attrs['winSize'] = _h5.none_parser(self.methodInfo['winSize'])
        h5group.attrs['overlap'] = _h5.none_parser(self.methodInfo['overlap'])
        self.systemSignal._h5_save(h5group.create_group('systemSignal'))
        pass

    def plot_time(self, *args, **kwargs):
        return self.systemSignal.plot_time(*args, **kwargs)

    def plot_time_dB(self, *args, **kwargs):
        return self.systemSignal.plot_time_dB(*args, **kwargs)

    def plot_freq(self, *args, **kwargs):
        return self.systemSignal.plot_freq(*args, **kwargs)

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
                data = _make_pk_spectra(inputSignal.freqSignal)
                outputFreqSignal = _make_pk_spectra(outputSignal.freqSignal)
                freqVector = inputSignal.freqVector
                b = data * 0 + 10**(-200/20) # inside signal freq range
                a = data * 0 + 1 # outside signal freq range
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
                        * float(np.max(np.abs(outputFreqSignal)))**2 \
                            * 1/2
                C = np.conj(data) / \
                    (np.conj(data)*data + eps)
                C = _make_rms_spectra(C)
                C = SignalObj(C,
                              'freq',
                              inputSignal.samplingRate,
                              signalType='energy')
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
                               samplingRate=inputSignal.samplingRate,
                               signalType='energy')
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
            result = SignalObj(samplingRate=inputSignal.samplingRate,
                               signalType='energy')
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
            result = SignalObj(samplingRate=inputSignal.samplingRate,
                               signalType='energy')
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
        return result

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


def _make_rms_spectra(freqSignal):
    newFreqSignal = np.zeros(freqSignal.shape, dtype=np.complex_)
    newFreqSignal[:,:] = freqSignal / 2**(1/2)
    newFreqSignal[0,:] = freqSignal[0,:] * 2**(1/2)
    return newFreqSignal


def _make_pk_spectra(freqSignal):
    newFreqSignal = np.zeros(freqSignal.shape, dtype=np.complex_)
    newFreqSignal[:,:] = freqSignal * 2**(1/2)
    newFreqSignal[0,:] = freqSignal[0,:] / 2**(1/2)
    return newFreqSignal

