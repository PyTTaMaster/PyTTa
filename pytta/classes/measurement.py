# -*- coding: utf-8 -*-

import os
import json
import zipfile
import numpy as np
import sounddevice as sd
import time
from . import _base
from .signal import SignalObj, ImpulsiveResponse


# Measurement class
class Measurement(_base.PyTTaObj):
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
        self.inChannel = _base.ChannelsList(inChannel)
        self.outChannel = _base.ChannelsList(outChannel)
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
        __print_max_level(recording, kind='input')
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
        __print_max_level(self.excitation, kind='output')
        __print_max_level(recording, kind='input')
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


# Sub functions
def __print_max_level(sigObj, kind):
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
