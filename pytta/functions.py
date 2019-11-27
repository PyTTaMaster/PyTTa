# -*- coding: utf-8 -*-
"""
Functions:
-----------

    This submodule carries a set of useful functions of general purpouses when
    using PyTTa, like reading and writing wave files, seeing the audio IO
    devices available and some signal processing tools.

    Available functions:
    ---------------------

        >>> pytta.list_devices()
        >>> pytta.fft_degree( timeLength, samplingRate)
        >>> pytta.read_wav( fileName )
        >>> pytta.write_wav( fileName, signalObject )
        >>> pytta.merge( signalObj1, signalObj2, ..., signalObjN )
        >>> pytta.fft_convolve( signalObj1, signalObj2 )
        >>> pytta.find_delay( signalObj1, signalObj2 )
        >>> pytta.corr_coef( signalObj1, signalObj2 )
        >>> pytta.resample( signalObj, newSamplingRate )
        >>> pytta.peak_time(signalObj1, signalObj2, ..., signalObjN )
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
from pytta.classes import SignalObj, ImpulsiveResponse, \
                    RecMeasure, PlayRecMeasure, FRFMeasure, \
                    Analysis
from pytta.classes._base import ChannelsList, ChannelObj
from pytta.generate import measurement  # TODO: Change to class instantiation.
import copy as cp
import pytta.h5utilities as _h5


def list_devices():
    """
    Shortcut to sounddevice.query_devices(). Made to exclude the need of
    importing Sounddevice directly just to find out which audio devices can be
    used.

        >>> pytta.list_devices()

    """
    return sd.query_devices()


def fft_degree(timeLength: float = 0, samplingRate: int = 1) -> float:
    """
    Returns the power of two value that can be used to calculate the total
    number of samples of the signal.

        >>> numSamples = 2**fftDegree

    Parameters:
    ------------

        timeLength (float = 0):
            Value, in seconds, of the time duration of the signal or
            recording.

        samplingRate (int = 1):
            Value, in samples per second, that the data will be captured
            or emitted.

    Returns:
    ---------

        fftDegree (float = 0):
            Power of 2 that can be used to calculate number of samples.

    """
    return np.log2(timeLength*samplingRate)



def read_wav(fileName):
    """
    Reads a wave file into a SignalObj
    """
    samplingRate, data = wf.read(fileName)
    if data.dtype == 'int16':
        data = data/(2**15)
    if data.dtype == 'int32':
        data = data/(2**31)
    signal = SignalObj(data, 'time', samplingRate=samplingRate)
    return signal


def write_wav(fileName, signalIn):
    """
    Writes a SignalObj into a single wave file
    """
    samplingRate = signalIn.samplingRate
    data = signalIn.timeSignal
    return wf.write(fileName, samplingRate, data)


# Refactor for new SignalObj's channelsList
def merge(signal1, *signalObjects):
    """
    Gather all of the input argument signalObjs into a single
    signalObj and place the respective timeSignal of each
    as a column of the new object
    """
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


# def split(signal):
#    return 0


def fft_convolve(signal1, signal2):
    """
    Uses scipy.signal.fftconvolve() to convolve two time domain signals.

        >>> convolution = pytta.fft_convolve(signal1,signal2)

    """
#    Fs = signal1.Fs
    conv = ss.fftconvolve(signal1.timeSignal, signal2.timeSignal)
    signal = SignalObj(conv, 'time', signal1.samplingRate)
    return signal


def find_delay(signal1, signal2):
    """
    Cross Correlation alternative, more efficient fft based method to calculate
    time shift between two signals.

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
    """
    Finds the correlation coeficient between two SignalObjs using
    the numpy.corrcoef() function.
    """
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


def save(fileName: str = time.ctime(time.time()), *PyTTaObjs):
    """
    Main save function for .pytta and .hdf5 files.

    The file format is choosed by the extension applied to the fileName. If no
    extension is provided choose the default file format.
    """
    # default file format
    defaultFormat = '.hdf5'
    # Checking the choosed file format
    if fileName.split('.')[-1] == 'hdf5':
        h5_save(fileName, *PyTTaObjs)
    elif fileName.split('.')[-1] == 'pytta':
        pytta_save(fileName, *PyTTaObjs)
    else:
        print('File extension must be .hdf5 or .pytta.\n'
        'Applying the default extension: {}.'.format(defaultFormat))
        fileName += defaultFormat
        save(fileName, *PyTTaObjs)


def load(fileName: str):
    """
    Main save function for .pytta and .hdf5 files.
    """
    if fileName.split('.')[-1] == 'hdf5':
        output = h5_load(fileName)
    elif fileName.split('.')[-1] == 'pytta':
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

    The .pytta extension must not be appended to the fileName
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


def h5_save(fileName: str, *PyTTaObjs):
    """
    Open an hdf5 file, create groups for each PyTTa object, pass it to
    the own object and it saves itself inside the group.

    >>> pytta.h5_save(fileName, PyTTaObj_1, PyTTaObj_2, ..., PyTTaObj_n)
    """
    # Checking if filename has .hdf5 extension
    if fileName.split('.')[-1] != 'hdf5':
        fileName += '.hdf5'
    with h5py.File(fileName, 'w') as f:
        # Dict for counting equal names for correctly renaming
        objsNameCount = {}
        objCount = 0  # Counter for loaded objects
        totCount = 0  # Counter for total groups
    
        for idx, pobj in enumerate(PyTTaObjs):
            totCount += 1
            if isinstance(pobj, (SignalObj,
                                 ImpulsiveResponse,
                                 RecMeasure,
                                 PlayRecMeasure,
                                 FRFMeasure,
                                 Analysis)):
                # Check if creation_name was already used
                creationName = pobj.creation_name
                if creationName in objsNameCount:
                    objsNameCount[creationName] += 1
                    creationName += '_' + str(objsNameCount[creationName])
                else:
                    objsNameCount[creationName] = 1
                # create obj's group
                ObjGroup = f.create_group(creationName)
                # save the obj inside its group
                pobj.h5_save(ObjGroup)
                objCount += 1
            else:
                print("Only PyTTa objects can be saved through this" +
                      "function. Skipping object number " + str(idx) + ".")
    # Final message
    plural1 = 's' if objCount > 1 else ''
    print('Saved {} PyTTa object'.format(objCount) + plural1 +
        ' of {}'.format(totCount) +
        ' inside the hdf5 file.')
    return fileName


def h5_load(fileName: str):
    """
    Load an hdf5 file and recreate the PyTTa objects.
    """
    # Checking if the file is an hdf5 file
    if fileName.split('.')[-1] != 'hdf5':
        raise ValueError("h5_load function only works with *.hdf5 files")
    f = h5py.File(fileName, 'r')
    loadedObjects = {}
    objCount = 0  # Counter for loaded objects
    totCount = 0  # Counter for total groups
    for PyTTaObjName, PyTTaObjGroup in f.items():
        totCount += 1
        try:
            loadedObjects[PyTTaObjName] = __h5_unpack(PyTTaObjGroup)
            objCount += 1
        except NotImplementedError:
            print('Skipping hdf5 group named {} as it '.format(PyTTaObjName) +
                  'isnt an PyTTa object group.')
    f.close()
    # Final message
    plural1 = 's' if objCount > 1 else ''
    plural2 = 's' if totCount > 1 else ''
    print('Imported {} PyTTa object-like group'.format(objCount) + plural1 +
          ' of {} group'.format(totCount) + plural2 +
          ' inside the hdf5 file.')
    return loadedObjects


def __h5_unpack(ObjGroup):
    """
    Unpack an HDF5 group into its respective PyTTa object
    """
    if ObjGroup.attrs['class'] == 'SignalObj':
        # PyTTaObj attrs unpacking
        samplingRate = ObjGroup.attrs['samplingRate']
        freqMin = _h5.none_parser(ObjGroup.attrs['freqMin'])
        freqMax = _h5.none_parser(ObjGroup.attrs['freqMax'])
        lengthDomain = ObjGroup.attrs['lengthDomain']
        comment = ObjGroup.attrs['comment']
        # SignalObj attr unpacking
        channels = eval(ObjGroup.attrs['channels'])
        # Creating and conforming SignalObj
        SigObj = SignalObj(signalArray=np.array(ObjGroup['timeSignal']),
                           domain='time',
                           samplingRate=samplingRate,
                           freqMin=freqMin,
                           freqMax=freqMax,
                           comment=comment)
        SigObj.channels = channels
        SigObj.lengthDomain = lengthDomain
        return SigObj

    elif ObjGroup.attrs['class'] == 'ImpulsiveResponse':
        systemSignal = __h5_unpack(ObjGroup['systemSignal'])
        method = ObjGroup.attrs['method']
        winType = ObjGroup.attrs['winType']
        winSize = ObjGroup.attrs['winSize']
        overlap = ObjGroup.attrs['overlap']
        IR = ImpulsiveResponse(method=method,
                               winType=winType,
                               winSize=winSize,
                               overlap=overlap,
                               ir=systemSignal)
        return IR

    elif ObjGroup.attrs['class'] == 'RecMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = ObjGroup.attrs['samplingRate']
        freqMin = _h5.none_parser(ObjGroup.attrs['freqMin'])
        freqMax = _h5.none_parser(ObjGroup.attrs['freqMax'])
        comment = ObjGroup.attrs['comment']
        lengthDomain = ObjGroup.attrs['lengthDomain']
        fftDegree = ObjGroup.attrs['fftDegree']
        timeLength = ObjGroup.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(ObjGroup.attrs['device'])
        inChannels = eval(ObjGroup.attrs['inChannels'])
        blocking = ObjGroup.attrs['blocking']
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

    elif ObjGroup.attrs['class'] == 'PlayRecMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = ObjGroup.attrs['samplingRate']
        freqMin = _h5.none_parser(ObjGroup.attrs['freqMin'])
        freqMax = _h5.none_parser(ObjGroup.attrs['freqMax'])
        comment = ObjGroup.attrs['comment']
        lengthDomain = ObjGroup.attrs['lengthDomain']
        fftDegree = ObjGroup.attrs['fftDegree']
        timeLength =ObjGroup.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(ObjGroup.attrs['device'])
        inChannels = eval(ObjGroup.attrs['inChannels'])
        outChannels = eval(ObjGroup.attrs['outChannels'])
        blocking = ObjGroup.attrs['blocking']
        # PlayRecMeasure attrs unpacking
        excitation = __h5_unpack(ObjGroup['excitation'])
        outputAmplification = ObjGroup.attrs['outputAmplification']
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

    elif ObjGroup.attrs['class'] == 'FRFMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = ObjGroup.attrs['samplingRate']
        freqMin = _h5.none_parser(ObjGroup.attrs['freqMin'])
        freqMax = _h5.none_parser(ObjGroup.attrs['freqMax'])
        comment = ObjGroup.attrs['comment']
        lengthDomain = ObjGroup.attrs['lengthDomain']
        fftDegree = ObjGroup.attrs['fftDegree']
        timeLength = ObjGroup.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(ObjGroup.attrs['device'])
        inChannels = eval(ObjGroup.attrs['inChannels'])
        outChannels = eval(ObjGroup.attrs['outChannels'])
        blocking = ObjGroup.attrs['blocking']
        # PlayRecMeasure attrs unpacking
        excitation = __h5_unpack(ObjGroup['excitation'])
        outputAmplification = ObjGroup.attrs['outputAmplification']
        # FRFMeasure attrs unpacking
        method = _h5.none_parser(ObjGroup.attrs['method'])
        winType = _h5.none_parser(ObjGroup.attrs['winType'])
        winSize = _h5.none_parser(ObjGroup.attrs['winSize'])
        overlap = _h5.none_parser(ObjGroup.attrs['overlap'])
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

    elif ObjGroup.attrs['class'] == 'Analysis':
        # Analysis attrs unpacking
        anType = _h5.attr_parser(ObjGroup.attrs['anType'])
        nthOct = _h5.attr_parser(ObjGroup.attrs['nthOct'])
        minBand = _h5.attr_parser(ObjGroup.attrs['minBand'])
        maxBand = _h5.attr_parser(ObjGroup.attrs['maxBand'])
        comment = _h5.attr_parser(ObjGroup.attrs['comment'])
        title = _h5.attr_parser(ObjGroup.attrs['title'])
        dataLabel = _h5.attr_parser(ObjGroup.attrs['dataLabel'])
        errorLabel = _h5.attr_parser(ObjGroup.attrs['errorLabel'])
        xLabel = _h5.attr_parser(ObjGroup.attrs['xLabel'])
        yLabel = _h5.attr_parser(ObjGroup.attrs['yLabel'])
        # Analysis data unpacking
        data = np.array(ObjGroup['data'])
        # If error in save moment was None no group was created for it
        if 'error' in ObjGroup:
            error = np.array(ObjGroup['error'])
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

    else:
        raise NotImplementedError
