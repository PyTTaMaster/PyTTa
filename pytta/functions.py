# -*- coding: utf-8 -*-
"""
Set of useful functions of general purpouses when using PyTTa.

Includes reading and writing wave files, seeing the audio IO
devices available and few signal processing tools.

Available functions:

    >>> pytta.list_devices()
    >>> pytta.read_wav(fileName)
    >>> pytta.write_wav(fileName, signalObject)
    >>> pytta.merge(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.slipt(signalObj)
    >>> pytta.fft_convolve(signalObj1, signalObj2)
    >>> pytta.find_delay(signalObj1, signalObj2)
    >>> pytta.corr_coef(signalObj1, signalObj2)
    >>> pytta.resample(signalObj, newSamplingRate)
    >>> pytta.peak_time(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_time(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_time_dB(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_freq(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_bars(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.save(fileName, obj1, ..., objN)
    >>> pytta.load(fileName)

For further information, check the function specific documentation.

"""

import os
import time
import json
from scipy.io import wavfile as wf
import scipy.io as sio
import numpy as np
import sounddevice as sd
import scipy.signal as ss
import scipy.fftpack as sfft
import zipfile as zf
import h5py
from typing import Union, List
from pytta.classes import SignalObj, ImpulsiveResponse, \
                    RecMeasure, PlayRecMeasure, FRFMeasure, \
                    Analysis, OctFilter
from pytta.classes._base import ChannelsList, ChannelObj
from pytta.generate import measurement  # TODO: Change to class instantiation.
from pytta import _h5utils as _h5
import copy as cp
from warnings import warn
from pytta import _plot as plot
# For backwards compatibility purposes. Planned to get out of here
from pytta.utils.maths import fft_degree as new_fft_degree


def list_devices():
    """
    Shortcut to sounddevice.query_devices().

    Made to exclude the need of importing Sounddevice directly
    just to find out which audio devices can be used.

        >>> pytta.list_devices()

    Returns
    -------
        A tuple containing all available audio devices.

    """
    return sd.query_devices()


def print_devices():
    """
    Print the devices list to stdout.

    Returns
    -------
    None.

    """
    return print(list_devices())


def get_device_from_user() -> Union[List[int], int]:
    """
    Print the device list and query for a number input of the device, or devices.

    Returns
    -------
    Union[List[int], int]
        Practical interface for querying devices to be used within scripts.

    """
    print_devices()
    device = [int(dev.strip()) for dev in input("Input the device number: ").split(',')]
    if len(device) == 1:
        device = device[0]
        text = "Device is:"
    else:
        text = "Devices are:"
    print(text, device)
    return device


def read_wav(fileName):
    """Read a wave file into a SignalObj."""
    samplingRate, data = wf.read(fileName)
    if data.dtype == 'int16':
        data = data/(2**15)
    if data.dtype == 'int32':
        data = data/(2**31)
    signal = SignalObj(data, 'time', samplingRate=samplingRate)
    return signal


def write_wav(fileName, signalIn):
    """Write a SignalObj into a single wave file."""
    samplingRate = signalIn.samplingRate
    data = signalIn.timeSignal
    return wf.write(fileName if '.wav' in fileName else fileName+'.wav', samplingRate, data)


def SPL(signal, nthOct=3, minFreq=100, maxFreq=4000):
    """
    Calculate the `signal`'s Sound Pressure Level

    The calculations are made by frequency bands and ranges from `minFreq` to
    `maxFreq` with `nthOct` bands per octave.

    Returns
    -------
        Analysis: The sound pressure level packed into an Analysis object.

    """
    with OctFilter(order=4, nthOct=nthOct, minFreq=minFreq, maxFreq=maxFreq,
                   base=10, refFreq=1000, samplingRate=signal.samplingRate) as ofb:
        fsignal = ofb.filter(signal)
    out = []
    for filtsignal in fsignal:
        out.append(Analysis('L', nthOct, minFreq, maxFreq, filtsignal.spl()))
    return out if len(out) > 1 else out[0]


def merge(signal1, *signalObjects):
    """Gather all channels of the signalObjs given as input arguments into a single SignalObj."""
    j = 1
    freqMin = cp.deepcopy(signal1.freqMin)
    freqMax = cp.deepcopy(signal1.freqMax)
    comment = cp.deepcopy(signal1.comment)
    channels = cp.deepcopy(signal1.channels)
    timeSignal = cp.deepcopy(signal1.timeSignal)
    for inObj in signalObjects:
        if signal1.samplingRate != inObj.samplingRate:
            message = '\
            \n To merge signals they must have the same sampling rate!\
            \n SignalObj 1 and '+str(j+1)+' have different sampling rates.'
            raise AttributeError(message)
        if signal1.numSamples != inObj.numSamples:
            message = '\
            \n To merge signals they must have the same length!\
            \n SignalObj 1 and '+str(j+1)+' have different lengths.'
            raise AttributeError(message)
        comment = comment + ' / ' + inObj.comment
        for ch in inObj.channels._channels:
            channels.append(ch)
        timeSignal = np.hstack((timeSignal, inObj.timeSignal))
        j += 1
    newSignal = SignalObj(timeSignal, domain='time',
                          samplingRate=signal1.samplingRate,
                          freqMin=freqMin, freqMax=freqMax, comment=comment)
    channels.conform_to()
    newSignal.channels = channels
    return newSignal


def split(*signalObjects,
          channels: list = None) -> list:
    """
    Split the provided SignalObjs' channels into several SignalObjs.

    If the 'channels' input argument is given, split the specified channel numbers of
    each SignalObj, otherwise split all channels.

    Arguments (default), (type):
    -----------------------------

        * non-keyword arguments (), (SignalObj)

        * channels (None), (list):
            specified channels to split from the provided SignalObjs;

    Return (type):
    --------------

        * spltdChs (list):
            a list containing SignalObjs for each split channel;

    """
    spltdChs = []

    for sigObj in signalObjects:
        moreSpltdChs = sigObj.split(channels=channels)
        spltdChs.extend(moreSpltdChs)

    return spltdChs


def fft_convolve(signal1, signal2):
    """
    Use scipy.signal.fftconvolve() to convolve two time domain signals.

        >>> convolution = pytta.fft_convolve(signal1,signal2)

    """
#    Fs = signal1.Fs
    conv = ss.fftconvolve(signal1.timeSignal, signal2.timeSignal)
    signal = SignalObj(conv, 'time', signal1.samplingRate)
    return signal


def find_delay(signal1, signal2):
    """
    Cross Correlation alternative.

    More efficient fft based method to calculate time shift between two signals.

        >>> shift = pytta.find_delay(signal1,signal2)

    """
    if signal1.N != signal2.N:
        return print('Signal1 and Signal2 must have the same length')
    else:
        freqSignal1 = signal1.freqSignal
        freqSignal2 = sfft.fft(np.flipud(signal2.timeSignal))
        convoluted = np.real(sfft.ifft(freqSignal1 * freqSignal2))
        convShifted = sfft.fftshift(convoluted)
        zeroIndex = int(signal1.numSamples / 2) - 1
        shift = zeroIndex - np.argmax(convShifted)
    return shift


def corr_coef(signal1, signal2):
    """Finds the correlation coefficient between two SignalObjs using the numpy.corrcoef() function."""
    coef = np.corrcoef(signal1.timeSignal, signal2.timeSignal)
    return coef[0, 1]


def resample(signal, newSamplingRate):
    """
    Resample the timeSignal of the input SignalObj to the
    given sample rate using the scipy.signal.resample() function
    """
    newSignalSize = np.int(signal.timeLength*newSamplingRate)
    resampled = ss.resample(signal.timeSignal[:], newSignalSize)
    newSignal = SignalObj(resampled, "time", newSamplingRate)
    return newSignal


def peak_time(signal):
    """
    Return the time at signal's amplitude peak.
    """
    if not isinstance(signal, SignalObj):
        raise TypeError('Signal must be an SignalObj.')
    peaks_time = []
    for chindex in range(signal.numChannels):
        maxamp = max(np.abs(signal.timeSignal[:, chindex]))
        maxindex = np.where(signal.timeSignal[:, chindex] == np.abs(maxamp))[0]
        maxtime = signal.timeVector[maxindex][0]
        peaks_time.append(maxtime)
    if signal.numChannels > 1:
        return peaks_time
    else:
        return peaks_time[0]

def fft_degree(*args,**kwargs):
    """
    DEPRECATED
    ----------
        Being replaced by pytta.utils.maths.fft_degree on version 0.1.0.

    Power-of-two value that can be used to calculate the total number of samples of the signal.

        >>> numSamples = 2**fftDegree

    Parameters
    ----------
        * timeLength (float = 0):
            Value, in seconds, of the time duration of the signal or
            recording.

        * samplingRate (int = 1):
            Value, in samples per second, that the data will be captured
            or emitted.

    Returns
    -------
        fftDegree (float = 0):
            Power of 2 that can be used to calculate number of samples.

    """
    warn(DeprecationWarning("Function 'pytta.fft_degree' is DEPRECATED and " +
                            "being replaced by pytta.utils.maths.fft_degree" +
                            " on version 0.1.0"))
    return new_fft_degree(*args, **kwargs)

def plot_time(*sigObjs, xLabel:str=None, yLabel:str=None, yLim:list=None,
              xLim:list=None, title:str=None, decimalSep:str=',',
              timeUnit:str='s'):
    """Plot provided SignalObjs togheter in time domain.

    Saves xLabel, yLabel, and title when provided for the next plots.

    Parameters (default), (type):
    -----------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

        * xLabel (None), (str):
            x axis label.

        * yLabel (None), (str):
            y axis label.

        * yLim (None), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (None), (str):
            left and right limits.

            >>> xLim = [0, 15]

        * title (None), (str):
            plot title.

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

        * timeUnit ('s'), (str):
            'ms' or 's'.

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    realSigObjs = _remove_non_(SignalObj, sigObjs, msgPrefix='plot_time:')
    if len(realSigObjs) > 0:
        fig = plot.time(realSigObjs, xLabel, yLabel, yLim, xLim, title,
                        decimalSep, timeUnit)
        return fig
    else:
        return

def plot_time_dB(*sigObjs, xLabel:str=None, yLabel:str=None, yLim:list=None,
              xLim:list=None, title:str=None, decimalSep:str=',',
              timeUnit:str='s'):
    """Plot provided SignalObjs togheter in decibels in time domain.

    Parameters (default), (type):
    -----------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

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
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_time_dB:')
    if len(realSigObjs) > 0:
        fig = plot.time_dB(realSigObjs, xLabel, yLabel, yLim, xLim, title,
                           decimalSep, timeUnit)
        return fig
    else:
        return


def plot_freq(*sigObjs, smooth:bool=False, xLabel:str=None, yLabel:str=None,
              yLim:list=None, xLim:list=None, title:str=None,
              decimalSep:str=','):
    """Plot provided SignalObjs magnitudes togheter in frequency domain.

    Parameters (default), (type):
    -----------------------------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

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
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_freq:')
    if len(realSigObjs) > 0:
        fig = plot.freq(realSigObjs, smooth, xLabel, yLabel, yLim, xLim, title,
                        decimalSep)
        return fig
    else:
        return

def plot_bars(*analyses, xLabel:str=None, yLabel:str=None,
              yLim:list=None, xLim:list=None, title:str=None, decimalSep:str=',',
              barWidth:float=0.75, errorStyle:str=None,
              forceZeroCentering:bool=False, overlapBars:bool=False,
              color:list=None):
    """Plot the analysis data in fractinal octave bands.

    Parameters (default), (type):
    -----------------------------

        * analyses (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            bands limits.

            >>> xLim = [100, 10000]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

        * barWidth (0.75), float:
            width of the bars from one fractional octave band.
            0 < barWidth < 1.

        * errorStyle ('standard'), str:
            error curve style. May be 'laza' or None/'standard'.

        * forceZeroCentering ('False'), bool:
            force centered bars at Y zero.

        * overlapBars ('False'), bool:
            overlap bars. No side by side bars of different data.

        * color (None), list:
            list containing the color of each Analysis.


    Return:
    --------

        matplotlib.figure.Figure object.
    """

    analyses = _remove_non_(Analysis, analyses, msgPrefix='plot_bars:')
    if len(analyses) > 0:
        fig = plot.bars(analyses, xLabel, yLabel, yLim, xLim, title,
            decimalSep, barWidth, errorStyle, forceZeroCentering, overlapBars,
            color)
        return fig
    else:
        return

def plot_spectrogram(*sigObjs, winType:str='hann', winSize:int=1024,
                     overlap:float=0.5, xLabel:str=None, yLabel:str=None,
                     yLim:list=None, xLim:list=None, title:str=None,
                     decimalSep:str=','):
    """
    Plots provided SignalObjs spectrogram.

    Parameters (default), (type):
    -----------------------------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

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
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_spectrogram:')
    if len(realSigObjs) > 0:
        figs = plot.spectrogram(realSigObjs, winType, winSize,
                                overlap, xLabel, yLabel, xLim, yLim,
                                title, decimalSep)
        return figs
    else:
        return

def plot_waterfall(*sigObjs, step=10, xLim:list=None,
                   Pmin=20, Pmax=None, tmin=0, tmax=None, azim=-72, elev=14,
                   cmap='jet', winPlot=False, waterfallPlot=True, fill=True,
                   lines=False, alpha=1, figsize=(20, 8), winAlpha=0,
                   removeGridLines=False, saveFig=False, bar=False, width=0.70,
                   size=3, lcol=None, filtered=True):
    """
    This function was gently sent by Rinaldi Polese Petrolli.

    # TO DO

    Keyword Arguments:
        step {int} -- [description] (default: {10})
        xLim {list} -- [description] (default: {None})
        Pmin {int} -- [description] (default: {20})
        Pmax {[type]} -- [description] (default: {None})
        tmin {int} -- [description] (default: {0})
        tmax {[type]} -- [description] (default: {None})
        azim {int} -- [description] (default: {-72})
        elev {int} -- [description] (default: {14})
        cmap {str} -- [description] (default: {'jet'})
        winPlot {bool} -- [description] (default: {False})
        waterfallPlot {bool} -- [description] (default: {True})
        fill {bool} -- [description] (default: {True})
        lines {bool} -- [description] (default: {False})
        alpha {int} -- [description] (default: {1})
        figsize {tuple} -- [description] (default: {(20, 8)})
        winAlpha {int} -- [description] (default: {0})
        removeGridLines {bool} -- [description] (default: {False})
        saveFig {bool} -- [description] (default: {False})
        bar {bool} -- [description] (default: {False})
        width {float} -- [description] (default: {0.70})
        size {int} -- [description] (default: {3})
        lcol {[type]} -- [description] (default: {None})
        filtered {bool} -- [description] (default: {True})

    Returns:
        [type] -- [description]
    """
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_waterfall:')
    if len(realSigObjs) > 0:
        figs = plot.waterfall(realSigObjs, step, xLim,
                              Pmin, Pmax, tmin, tmax, azim, elev,
                              cmap, winPlot, waterfallPlot, fill,
                              lines, alpha, figsize, winAlpha,
                              removeGridLines, saveFig, bar, width,
                              size, lcol, filtered)
        return figs
    else:
        return

def _remove_non_(dataType, dataSet,
                 msgPrefix:str='_remove_non_SignalObjs:'):
    if isinstance(dataSet, (list, tuple)):
        newDataSet = []
        for idx, item in enumerate(dataSet):
            if isinstance(item, dataType):
                newDataSet.append(item)
            elif isinstance(item, ImpulsiveResponse) and \
                dataType.__name__ == 'SignalObj':
                newDataSet.append(item.systemSignal)
            else:
                print("{}: skipping object {} as it isn't a {}."
                      .format(msgPrefix, idx+1, dataType.__name__))
        if isinstance(dataSet, tuple):
            newDataSet = tuple(newDataSet)
    return newDataSet

def save(fileName: str = time.ctime(time.time()), *PyTTaObjs):
    """
    Main save function for .hdf5 and .pytta files.

    The file format is chose by the extension applied to the fileName. If no
    extension is provided choose the default file format (.hdf5).

    For more information on saving PyTTa objects in .hdf5 format see
    pytta.functions._h5_save documentation.

    For more information on saving PyTTa objects in .pytta format see
    pytta.functions.pytta_save' documentation. (DEPRECATED)
    """
    # default file format
    defaultFormat = '.hdf5'
    # Checking the chosen file format
    if fileName.split('.')[-1] == 'hdf5':
        _h5_save(fileName, *PyTTaObjs)
    elif fileName.split('.')[-1] == 'pytta': # DEPRECATED
        warn(DeprecationWarning("'.pytta' format is DEPRECATED and being " +
                                "replaced by '.hdf5'."))
        pytta_save(fileName, *PyTTaObjs)
    else:
        print("File extension must be '.hdf5'.\n" +
              "Applying the default extension.")
        fileName += defaultFormat
        save(fileName, *PyTTaObjs)


def load(fileName: str):
    """
    Main save function for .pytta and .hdf5 files.
    """
    if fileName.split('.')[-1] == 'hdf5':
        output = _h5_load(fileName)
    elif fileName.split('.')[-1] == 'pytta':
        warn(DeprecationWarning("'.pytta' format is DEPRECATED and being " +
                                "replaced by '.hdf5'."))
        output = pytta_load(fileName)
    else:
        ValueError('pytta.load only works with *.hdf5 or *.pytta files.')
    return output

def pytta_save(fileName: str = time.ctime(time.time()), *PyTTaObjs):
    """
    Saves any number of PyTTaObj subclasses' objects to fileName.pytta file.

    Just calls .save() method of each class and packs them all into a major
    .pytta file along with a Meta.json file containing the fileName of each
    saved object.
    """
    if fileName.split('.')[-1] == 'pytta':
        fileName = fileName.replace('.pytta', '')
    meta = {}
    with zf.ZipFile(fileName + '.pytta', 'w') as zdir:
        for idx, obj in enumerate(PyTTaObjs):
            sobj = obj.pytta_save('obj' + str(idx))
            meta['obj' + str(idx)] = sobj
            zdir.write(sobj)
            os.remove(sobj)
        with open('Meta.json', 'w') as f:
            json.dump(meta, f, indent=4)
        zdir.write('Meta.json')
        os.remove('Meta.json')
    return fileName + '.pytta'


def pytta_load(fileName: str):
    """
    Loads .pytta files and parses it's types to the correct objects.
    """
    if fileName.split('.')[-1] == 'pytta':
        with zf.ZipFile(fileName, 'r') as zdir:
            objects = zdir.namelist()
            for obj in objects:
                if obj.split('.')[-1] == 'json':
                    meta = obj
            zdir.extractall()
            output = __parse_load(meta)
    else:
        raise ValueError("pytta_load function only works with *.pytta files")
    return output


def __parse_load(className):
    name = className.split('.')[0]
    jsonFile = open(className, 'r')
    openJson = json.load(jsonFile)
    if name == 'SignalObj':
        openMat = sio.loadmat(openJson['timeSignalAddress'])
        out = SignalObj(openMat['timeSignal'], domain=openJson['lengthDomain'],
                        samplingRate=openJson['samplingRate'],
                        freqMin=openJson['freqLims'][0],
                        freqMax=openJson['freqLims'][1],
                        comment=openJson['comment'])
        out.channels = __parse_channels(openJson['channels'],
                                        out.channels)
        os.remove(openJson['timeSignalAddress'])

    elif name == 'ImpulsiveResponse':
        ir = pytta_load(openJson['SignalAddress']['ir'])
        out = ImpulsiveResponse(ir=ir, **openJson['methodInfo'])
        os.remove(openJson['SignalAddress']['ir'])

    elif name == 'RecMeasure':
        inch = list(np.arange(len(openJson['inChannels'])))
        out = RecMeasure(device=openJson['device'], inChannels=inch,
                         lengthDomain='samples',
                         fftDegree=openJson['fftDegree'])
        out.inChannels = __parse_channels(openJson['inChannels'],
                                          out.inChannels)

    elif name == 'PlayRecMeasure':
        inch = list(1 + np.arange(len(openJson['inChannels'])))
        excit = pytta_load(openJson['excitationAddress'])
        out = PlayRecMeasure(excitation=excit,
                             device=openJson['device'], inChannels=inch)
        out.inChannels = __parse_channels(openJson['inChannels'],
                                          out.inChannels)
        os.remove(openJson['excitationAddress'])

    elif name == 'FRFMeasure':
        inch = list(1 + np.arange(len(openJson['inChannels'])))
        excit = pytta_load(openJson['excitationAddress'])
        out = FRFMeasure(excitation=excit, device=openJson['device'],
                         inChannels=inch)
        out.inChannels = __parse_channels(openJson['inChannels'],
                                          out.inChannels)
        os.remove(openJson['excitationAddress'])

    elif name == 'Meta':
        out = []
        for val in openJson.values():
            out.append(pytta_load(val))
            os.remove(val)
    os.remove(className)
    jsonFile.close()
    return out


def __parse_channels(chDict, chList):
    ch = 1
    for key in chDict.keys():
        chList[ch].num = key
        chList[ch].unit = chDict[key]['unit']
        chList[ch].name = chDict[key]['name']
        chList[ch].CF = chDict[key]['calib'][0]
        chList[ch].calibCheck\
            = chDict[key]['calib'][1]
        ch += 1
    return chList


def _h5_save(fileName: str, *PyTTaObjs):
    """
    Open an hdf5 file, create groups for each PyTTa object, pass it to
    the own object and it saves itself inside the group.

    >>> pytta._h5_save(fileName, PyTTaObj_1, PyTTaObj_2, ..., PyTTaObj_n)

    Dictionaries can also be passed as a PyTTa object. An hdf5 group will be
    created for each dictionary and its PyTTa objects will be saved. To ensure
    the diciontary name will be saved, create the key 'dictName' inside it with
    its name in a string as the value. This function will take this key and use
    as variable name for the dict.

    Lists can also be passed as a PyTTa object. An hdf5 group will be created
    for each list and its PyTTa objects will be saved. To ensure the list name
    will be saved, append to the list a string containing its name. This
    function will take the first string found and use it as variable name for
    the list.
    """
    # Checking if filename has .hdf5 extension
    if fileName.split('.')[-1] != 'hdf5':
        fileName += '.hdf5'
    with h5py.File(fileName, 'w') as f:
        # Save the version to the HDF5 file
        f.attrs['GENERATED_BY'] = 'PyTTa'
        f.attrs['LONG_DESCR'] = 'HDF5 file generated by the PyTTa toolbox'
        f.attrs['FILE_SYS_VERSION'] = 1
        # Dict for counting equal names for correctly renaming
        totalPObjCount = 0  # Counter for total groups
        savedPObjCount = 0  # Counter for loaded objects
        for idx, pObj in enumerate(PyTTaObjs):
            packTotalPObjCount, packSavedPObjCount = \
                __h5_pack(f, pObj, idx)
            totalPObjCount, savedPObjCount = \
                totalPObjCount + packTotalPObjCount, \
                    savedPObjCount + packSavedPObjCount
    # Final message
    plural1 = 's' if savedPObjCount > 1 else ''
    plural2 = 's' if totalPObjCount > 1 else ''
    print('Saved inside the hdf5 file {} PyTTa object{}'
          .format(savedPObjCount, plural1) +
          ' of {} object{} provided.'.format(totalPObjCount, plural2))
    return fileName


def __h5_pack(rootH5Group, pObj, objDesc):
    """
    __h5_pack packs a PyTTa object or dict into its respective HDF5 group.
    """
    if isinstance(pObj, (SignalObj,
                         ImpulsiveResponse,
                         RecMeasure,
                         PlayRecMeasure,
                         FRFMeasure,
                         Analysis)):
        # Creation name
        if isinstance(objDesc, str):
            creationName = objDesc
        else:
            creationName = pObj.creation_name
        # Check if creation_name was already used
        creationName = __h5_pack_count_and_rename(creationName, rootH5Group)
        # create obj's group
        objH5Group = rootH5Group.create_group(creationName)
        # save the obj inside its group
        pObj._h5_save(objH5Group)
        return (1, 1)

    elif isinstance(pObj, dict):
        # Creation name
        if 'dictName' in pObj:
            creationName = pObj.pop('dictName')
        elif isinstance(objDesc, str):
            creationName = objDesc
        else:
            creationName = 'noNameDict'
        creationName = __h5_pack_count_and_rename(creationName, rootH5Group)
        print("Saving the dict '{}'.".format(creationName))
        # create obj's group
        objH5Group = rootH5Group.create_group(creationName)
        objH5Group.attrs['class'] = 'dict'
        # Saving each key of the dict inside the hdf5 group
        totalPObjCount = 0
        savedPObjCount = 0
        for key, pObjFromDict in pObj.items():
            packTotalPObjCount, packSavedPObjCount = \
                __h5_pack(objH5Group, pObjFromDict, key)
            totalPObjCount, savedPObjCount = \
                totalPObjCount + packTotalPObjCount, \
                    savedPObjCount + packSavedPObjCount
        return (totalPObjCount, savedPObjCount)

    elif isinstance(pObj, list):
        # Creation name
        creationName = None
        for idx, item in enumerate(pObj):
            if isinstance(item, str):
                creationName = item
                pObj.pop(idx)
                continue
        if creationName is None:
            if isinstance(objDesc, str):
                creationName = objDesc
            else:
                creationName = 'noNameList'
        creationName = __h5_pack_count_and_rename(creationName, rootH5Group)
        print("Saving the list '{}'.".format(creationName))
        # create obj's group
        objH5Group = rootH5Group.create_group(creationName)
        objH5Group.attrs['class'] = 'list'
        # Saving each item of the list inside the hdf5 group
        totalPObjCount = 0
        savedPObjCount = 0
        for idx, pObjFromList in enumerate(pObj):
            packTotalPObjCount, packSavedPObjCount = \
                __h5_pack(objH5Group, pObjFromList, str(idx))
            totalPObjCount, savedPObjCount = \
                totalPObjCount + packTotalPObjCount, \
                    savedPObjCount + packSavedPObjCount
        return totalPObjCount, savedPObjCount

    else:
        print("Only PyTTa objects and dicts/lists with PyTTa objects " +
              "can be saved through this function. Skipping " +
              "object '" + str(objDesc) + "'.")
        return (1, 0)


def __h5_pack_count_and_rename(creationName, h5Group):
    # Check if creation_name was already used
    objNameCount = 1
    newCreationName = cp.copy(creationName)
    while newCreationName in h5Group:
        objNameCount += 1
        newCreationName = \
            creationName + '_' + str(objNameCount)
    creationName = newCreationName
    return creationName


def _h5_load(fileName: str):
    """
    Load an hdf5 file and recreate the PyTTa objects.
    """
    # Checking if the file is an hdf5 file
    if fileName.split('.')[-1] != 'hdf5':
        raise ValueError("_h5_load function only works with *.hdf5 files")
    f = h5py.File(fileName, 'r')

    # Check if it is a PyTTa-like hdf5 file
    try:
        if 'GENERATED_BY' not in f.attrs.keys() or  \
                                        f.attrs['GENERATED_BY'] != "PyTTa":
            raise NotImplementedError
    except:
        # raise NotImplementedError("Only PyTTa-like hdf5 files can be loaded.")
        warn(DeprecationWarning("'GENERATED_BY' tag couldn't be found in " +
                                "the .hdf5 file. Still trying to load " +
                                "because of legacy PyTTa HDF5 files."))

    loadedObjects = {}
    objCount = 0  # Counter for loaded objects
    totCount = 0  # Counter for total groups
    for PyTTaObjName, PyTTaObjH5Group in f.items():
        totCount += 1
        try:
            loadedObjects[PyTTaObjName] = __h5_unpack(PyTTaObjH5Group)
            objCount += 1
        except NotImplementedError:
            print("Skipping hdf5 group named {} as it ".format(PyTTaObjName) +
                  "isn't an PyTTa object group.")
    f.close()
    # Final message
    plural1 = 's' if objCount > 1 else ''
    plural2 = 's' if totCount > 1 else ''
    print('Imported {} PyTTa object-like group'.format(objCount) + plural1 +
          ' of {} group'.format(totCount) + plural2 +
          ' inside the hdf5 file.')
    return loadedObjects


def __h5_unpack(objH5Group):
    """
    Unpack an HDF5 group into its respective PyTTa object
    """
    if objH5Group.attrs['class'] == 'SignalObj':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        lengthDomain = objH5Group.attrs['lengthDomain']
        comment = objH5Group.attrs['comment']
        # SignalObj attr unpacking
        channels = eval(objH5Group.attrs['channels'])
        # Added with an if for compatibility issues
        if 'signalType' in objH5Group.attrs:
            signalType = _h5.attr_parser(objH5Group.attrs['signalType'])
        else:
            signalType = 'power'
        # Creating and conforming SignalObj
        SigObj = SignalObj(signalArray=np.array(objH5Group['timeSignal']),
                           domain='time',
                           signalType=signalType,
                           samplingRate=samplingRate,
                           freqMin=freqMin,
                           freqMax=freqMax,
                           comment=comment)
        SigObj.channels = channels
        SigObj.lengthDomain = lengthDomain
        return SigObj

    elif objH5Group.attrs['class'] == 'ImpulsiveResponse':
        systemSignal = __h5_unpack(objH5Group['systemSignal'])
        method = objH5Group.attrs['method']
        winType = objH5Group.attrs['winType']
        winSize = objH5Group.attrs['winSize']
        overlap = objH5Group.attrs['overlap']
        IR = ImpulsiveResponse(method=method,
                               winType=winType,
                               winSize=winSize,
                               overlap=overlap,
                               ir=systemSignal)
        return IR

    elif objH5Group.attrs['class'] == 'RecMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        comment = objH5Group.attrs['comment']
        lengthDomain = objH5Group.attrs['lengthDomain']
        fftDegree = objH5Group.attrs['fftDegree']
        timeLength = objH5Group.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(objH5Group.attrs['device'])
        inChannels = eval(objH5Group.attrs['inChannels'])
        blocking = objH5Group.attrs['blocking']
        # Recreating the object
        rObj = measurement(kind='rec',
                           device=device,
                           inChannels=inChannels,
                           blocking=blocking,
                           samplingRate=samplingRate,
                           freqMin=freqMin,
                           freqMax=freqMax,
                           comment=comment,
                           lengthDomain=lengthDomain,
                           fftDegree=fftDegree,
                           timeLength=timeLength)
        return rObj

    elif objH5Group.attrs['class'] == 'PlayRecMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        comment = objH5Group.attrs['comment']
        lengthDomain = objH5Group.attrs['lengthDomain']
        fftDegree = objH5Group.attrs['fftDegree']
        timeLength =objH5Group.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(objH5Group.attrs['device'])
        inChannels = eval(objH5Group.attrs['inChannels'])
        outChannels = eval(objH5Group.attrs['outChannels'])
        blocking = objH5Group.attrs['blocking']
        # PlayRecMeasure attrs unpacking
        excitation = __h5_unpack(objH5Group['excitation'])
        outputAmplification = objH5Group.attrs['outputAmplification']
        # Recreating the object
        prObj = measurement(kind='playrec',
                            excitation=excitation,
                            outputAmplification=outputAmplification,
                            device=device,
                            inChannels=inChannels,
                            outChannels=outChannels,
                            blocking=blocking,
                            samplingRate=samplingRate,
                            freqMin=freqMin,
                            freqMax=freqMax,
                            comment=comment)
        return prObj

    elif objH5Group.attrs['class'] == 'FRFMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        comment = objH5Group.attrs['comment']
        lengthDomain = objH5Group.attrs['lengthDomain']
        fftDegree = objH5Group.attrs['fftDegree']
        timeLength = objH5Group.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(objH5Group.attrs['device'])
        inChannels = eval(objH5Group.attrs['inChannels'])
        outChannels = eval(objH5Group.attrs['outChannels'])
        blocking = objH5Group.attrs['blocking']
        # PlayRecMeasure attrs unpacking
        excitation = __h5_unpack(objH5Group['excitation'])
        outputAmplification = objH5Group.attrs['outputAmplification']
        # FRFMeasure attrs unpacking
        method = _h5.none_parser(objH5Group.attrs['method'])
        winType = _h5.none_parser(objH5Group.attrs['winType'])
        winSize = _h5.none_parser(objH5Group.attrs['winSize'])
        overlap = _h5.none_parser(objH5Group.attrs['overlap'])
        # Recreating the object
        frfObj = measurement(kind='frf',
                             method=method,
                             winType=winType,
                             winSize=winSize,
                             overlap=overlap,
                             excitation=excitation,
                             outputAmplification=outputAmplification,
                             device=device,
                             inChannels=inChannels,
                             outChannels=outChannels,
                             blocking=blocking,
                             samplingRate=samplingRate,
                             freqMin=freqMin,
                             freqMax=freqMax,
                             comment=comment)
        return frfObj

    elif objH5Group.attrs['class'] == 'Analysis':
        # Analysis attrs unpacking
        anType = _h5.attr_parser(objH5Group.attrs['anType'])
        nthOct = _h5.attr_parser(objH5Group.attrs['nthOct'])
        minBand = _h5.attr_parser(objH5Group.attrs['minBand'])
        maxBand = _h5.attr_parser(objH5Group.attrs['maxBand'])
        comment = _h5.attr_parser(objH5Group.attrs['comment'])
        title = _h5.attr_parser(objH5Group.attrs['title'])
        dataLabel = _h5.attr_parser(objH5Group.attrs['dataLabel'])
        errorLabel = _h5.attr_parser(objH5Group.attrs['errorLabel'])
        xLabel = _h5.attr_parser(objH5Group.attrs['xLabel'])
        yLabel = _h5.attr_parser(objH5Group.attrs['yLabel'])
        # Analysis data unpacking
        data = np.array(objH5Group['data'])
        # If error in save moment was None no group was created for it
        if 'error' in objH5Group:
            error = np.array(objH5Group['error'])
        else:
            error = None
        # Recreating the object
        anObject = Analysis(anType=anType,
                            nthOct=nthOct,
                            minBand=minBand,
                            maxBand=maxBand,
                            data=data,
                            dataLabel=dataLabel,
                            error=error,
                            errorLabel=errorLabel,
                            comment=comment,
                            xLabel=xLabel,
                            yLabel=yLabel,
                            title=title)
        return anObject

    elif objH5Group.attrs['class'] == 'dict':
        dictObj = {}
        for PyTTaObjName, PyTTaObjH5Group in objH5Group.items():
            dictObj[PyTTaObjName] = __h5_unpack(PyTTaObjH5Group)
        return dictObj

    elif objH5Group.attrs['class']  == 'list':
        dictObj = {}
        for idx, PyTTaObjH5Group in objH5Group.items():
            dictObj[int(idx)] = __h5_unpack(PyTTaObjH5Group)
        idxs = [int(item) for item in list(dictObj.keys())]
        maxIdx = max(idxs)
        listObj = []
        for idx in range(maxIdx+1):
            listObj.append(dictObj[idx])
        return listObj

    else:
        raise NotImplementedError
