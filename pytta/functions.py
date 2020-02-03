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
        >>> pytta.fft_degree(timeLength, samplingRate)
        >>> pytta.read_wav(fileName)
        >>> pytta.write_wav(fileName, signalObject)
        >>> pytta.merge(signalObj1, signalObj2, ..., signalObjN)
        >>> pytta.fft_convolve(signalObj1, signalObj2)
        >>> pytta.find_delay(signalObj1, signalObj2)
        >>> pytta.corr_coef(signalObj1, signalObj2)
        >>> pytta.resample(signalObj, newSamplingRate)
        >>> pytta.peak_time(signalObj1, signalObj2, ..., signalObjN)
        >>> pytta.save(fileName, obj1, ..., objN)
        >>> pytta.load(fileName)
        >>> pytta.plot_time(signalObj1, signalObj2, ..., signalObjN)


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
from pytta import h5utils as _h5
import copy as cp
from warnings import warn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale


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

    For more information on saving PyTTa objects in .hdf5 format see
    pytta.functions.h5_save' documentation.

    For more information on saving PyTTa objects in .pytta format see
    pytta.functions.pytta_save' documentation. (DEPRECATED)
    """
    # default file format
    defaultFormat = '.hdf5'
    # Checking the choosed file format
    if fileName.split('.')[-1] == 'hdf5':
        h5_save(fileName, *PyTTaObjs)
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
        output = h5_load(fileName)
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


def h5_save(fileName: str, *PyTTaObjs):
    """
    Open an hdf5 file, create groups for each PyTTa object, pass it to
    the own object and it saves itself inside the group.

    >>> pytta.h5_save(fileName, PyTTaObj_1, PyTTaObj_2, ..., PyTTaObj_n)

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
        pObj.h5_save(objH5Group)
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
    for PyTTaObjName, PyTTaObjH5Group in f.items():
        totCount += 1
        try:
            loadedObjects[PyTTaObjName] = __h5_unpack(PyTTaObjH5Group)
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
        # SignalObj attr unpacking
        channels = eval(objH5Group.attrs['channels'])
        # Added with an if for compatibilitie issues
        if 'signalType' in objH5Group.attrs:
            signalType = _h5.attr_parser(objH5Group.attrs['signalType'])
        else:
            signalType = 'power'
        comment = objH5Group.attrs['comment']
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

def plot_time(*SigObjs, xLabel:str=None, yLabel:str=None, yLim:list=None,
              xLim:list=None, title:str=None,
              decimalSep:str=','):
    """plot_time plot one or more SignalObj together
    
    :param SigObjs: SignalObjs as arguments to be ploted
    :type SigObjs: SignalObj
    :param xLabel: x axis label, defaults to None
    :type xLabel: str, optional
    :param yLabel: y axis label, defaults to None
    :type yLabel: str, optional
    :param yLim: y axis limits, defaults to None
    :type yLim: list, optional
    :param title: plot title, defaults to None
    :type title: str, optional
    :param decimalSep: decimal separator, defaults to ','
    :type decimalSep: str, optional
    """
    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Time [s]'
    
    if yLabel is None:
        yLabel = 'Amplitude'
    
    if title is None:
        title = ''

    xLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)
    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)

    for SigObj in SigObjs:

        for chIndex in range(SigObj.numChannels):
            chNum = SigObj.channels.mapping[chIndex]
            label = '{} [{}]'.format(SigObj.channels[chNum].name,
                                     SigObj.channels[chNum].unit)
            ax.plot(SigObj.timeVector,
                    SigObj.timeSignal[:, chIndex],
                    label=label)

        xLims = \
            np.vstack((xLims, [SigObj.timeVector[0], SigObj.timeVector[-1]]))

        yLimData = SigObj.timeSignal[:, chIndex]
        yLimData[np.isinf(yLimData)] = 0    
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))
    
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    ax.legend(loc='best', fontsize=12)

    if xLim is None:
        xLim = [np.nanmin(xLims[:,0]), np.nanmax(xLims[:,1])]
    ax.set_xlim(xLim)

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
        margin = (yLim[1] - yLim[0]) / 20
        yLim = [yLim[0]-margin, yLim[1]+margin]
    ax.set_ylim(yLim)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    return fig
