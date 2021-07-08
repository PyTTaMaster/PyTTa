# -*- coding: utf-8 -*-

import os
import json
import zipfile
import numpy as np
import sounddevice as sd
import time
from pytta.classes import _base
from pytta.classes.streaming import Monitor, Streaming
from pytta.classes.signal import SignalObj, ImpulsiveResponse
from pytta import _h5utils as _h5
import traceback


# Measurement class
class _MeasurementBase(_base.PyTTaObj):
    """
    Measurement object class created to define some properties and methods to
    be used by the playback, recording and processing classes. It is a private
    class

    Properties(self): (default), (dtype), meaning;

        * device (system default), (list/int):
            list of input and output devices;

        * inChannels ([1]), (ChannelsList | list[int]):
            list of device's input channel used for recording;

        * outChannels ([1]), (ChannelsList | list[int]):
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
            2**fftDegree signal's number of samples for
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
                 inChannels=None,
                 outChannels=None,
                 blocking=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # device number. For device list use sounddevice.query_devices()
        self.device = device
        self.inChannels = _base.ChannelsList(inChannels)
        self.outChannels = _base.ChannelsList(outChannels)
        self.blocking = blocking
        return

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                # Measurement properties
                f'device={self.device!r}, '
                f'inChannels={self.inChannels!r}, '
                f'outChannels={self.outChannels!r}, '
                f'blocking={self.blocking!r}, '
                # PyTTaObj properties
                f'samplingRate={self.samplingRate!r}, '
                f'freqMin={self.freqMin!r}, '
                f'freqMax={self.freqMax!r}, '
                f'comment={self.comment!r}), '
                f'lengthDomain={self.lengthDomain!r}, '
                f'fftDegree={self.fftDegree!r}, '
                f'timeLength={self.timeLength!r}')

    def _to_dict(self):
        out = {'device': self.device,
               'inChannels': self.inChannels._to_dict(),
               'outChannels': self.outChannels._to_dict()}
        return out

    def _h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already opened file.
        """
        h5group.attrs['device'] = _h5.list_w_int_parser(self.device)
        h5group.attrs['inChannels'] = repr(self.inChannels)
        h5group.attrs['outChannels'] = repr(self.outChannels)
        h5group.attrs['blocking'] = self.blocking
        super()._h5_save(h5group)
        pass

# Measurement Properties
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, newDevice):
        self._device = newDevice
        return

    @property
    def numInChannels(self):
        return len(self.inChannels)

    @property
    def numOutChannels(self):
        return len(self.outChannels)

    def calib_pressure(self, chIndex,
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

        refSignalObj = RecMeasure(lengthDomain='time',
                                  timeLength=5,
                                  samplingRate=self.samplingRate,
                                  inChannels=chIndex,
                                  device=self.device,
                                  freqMin=self.freqMin,
                                  freqMax=self.freqMax).run()
        if chIndex-1 in range(len(self.inChannels)):
            self.inChannels[chIndex-1].calib_press(refSignalObj,
                                                   refPrms, refFreq)
            self.inChannels[chIndex-1].calibCheck = True
        else:
            raise IndexError('chIndex not in list of channel numbers')
        return


class Measurement(_MeasurementBase):
    """
    Measurement object class created to define some properties and methods to
    be used by the playback, recording and processing classes. It is a private
    class

    Properties(self): (default), (dtype), meaning;

        * device (system default), (list/int):
            list of input and output devices;

        * inChannels ([1]), (ChannelsList | list[int]):
            list of device's input channel used for recording;

        * outChannels ([1]), (ChannelsList | list[int]):
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
            2**fftDegree signal's number of samples for
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
                 inChannels=None,
                 outChannels=None,
                 blocking=True,
                 *args,
                 **kwargs):
        super().__init__(device, inChannels, outChannels, blocking, *args, **kwargs)
        return

    def play(self, monitor = Monitor(5512)):
        """
        Play SignalObj.

        Returns
        -------
        None.

        """
        with Streaming('O',
                       self.samplingRate,
                       self.device,
                       'float32',
                       0,
                       None,
                       self.outChannels,
                       self.excitation,
                       None,
                       self.numSamples,
                       monitor) as strm:
            strm.play()
        return

    def record(self, monitor = Monitor(5512)):
        with Streaming('I',
                       self.samplingRate,
                       self.device,
                       'float32',
                       0,
                       self.inChannels,
                       None,
                       None,
                       None,
                       self.numSamples,
                       monitor) as strm:
            rec = strm.record()
        return SignalObj(rec, 'time', self.samplingRate)

    def playrec(self, monitor = Monitor(5512)):
        with Streaming('IO',
                       self.samplingRate,
                       self.device,
                       'float32',
                       0,
                       self.inChannels,
                       self.outChannels,
                       self.excitation,
                       None,
                       self.numSamples,
                       monitor) as strm:
            rec = strm.playrec()
        return SignalObj(rec, 'time', self.samplingRate)


# RecMeasure class
class RecMeasure(_MeasurementBase):
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

        * inChannels ([1]), (list/int):
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
            self.fftDegree = fftDegree
        elif self.lengthDomain == 'time':
            self.timeLength = timeLength
        else:
            self._timeLength = None
            self._fftDegree = None
        self._outChannels = None
        return

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                # RecMeasure properties
                f'lengthDomain={self.lengthDomain!r}, '
                f'fftDegree={self.fftDegree!r}, '
                f'timeLength={self.timeLength!r}, '
                # Measurement properties
                f'device={self.device!r}, '
                f'inChannels={self.inChannels!r}, '
                f'blocking={self.blocking!r}, '
                # PyTTaObj properties
                f'samplingRate={self.samplingRate!r}, '
                f'freqMin={self.freqMin!r}, '
                f'freqMax={self.freqMax!r}, '
                f'comment={self.comment!r})')

    def _to_dict(self):
        sup = super()._to_dict()
        sup['fftDegree'] = self.fftDegree
        return sup

    def pytta_save(self, dirname=time.ctime(time.time())):
        dic = self._to_dict()
        name = dirname + '.pytta'
        with zipfile.ZipFile(name, 'w') as zdir:
            with open('RecMeasure.json', 'w') as f:
                json.dump(dic, f, indent=4)
            zdir.write('RecMeasure.json')
        return name

    def _h5_save(self, h5group, setClass=True):
        """
        Saves itself inside a hdf5 group from an already openned file via
        pytta._h5_save(...). Use setClass=True if the attribute 'class' must be
        seted to RecMeasure.

        >>> RecMeasure._h5_save(h5group, setClass=True)

        """
        if setClass is True:
            h5group.attrs['class'] = 'RecMeasure'
        h5group.attrs['lengthDomain'] = _h5.none_parser(self.lengthDomain)
        h5group.attrs['fftDegree'] = _h5.none_parser(self.fftDegree)
        h5group.attrs['timeLength'] = _h5.none_parser(self.timeLength)
        super()._h5_save(h5group)
        pass

# Rec Properties
    @property
    def timeLength(self):
        return self._timeLength

    @timeLength.setter
    def timeLength(self, newLength):
        self._timeLength = np.round(newLength, 2)
        self._numSamples = int(self.timeLength * self.samplingRate)
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
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        # Record
        recording = sd.rec(frames=self.numSamples,
                           samplerate=self.samplingRate,
                           mapping=self.inChannels.mapping,
                           blocking=self.blocking,
                           device=self.device,
                           latency='low',
                           dtype='float32')
        recording = np.squeeze(recording)
        recording = SignalObj(signalArray=recording*self.inChannels.CFlist(),
                              domain='time',
                              samplingRate=self.samplingRate)
        recording.channels = self.inChannels
        recording.timeStamp = time.ctime(time.time())
        recording.freqMin, recording.freqMax\
            = self.freqMin, self.freqMax
        recording.comment = 'SignalObj from a Rec measurement'
        recording.creation_name = creation_name
        _print_max_level(recording, kind='input')
        return recording


# PlayRecMeasure class
class PlayRecMeasure(_MeasurementBase):
    """
    Playback and Record object

    Properties:
    ------------

        * excitation (SignalObj), (SignalObj):
            signal information used to reproduce (playback);

        * outputAmplification (0), (float):
            Gain in dB applied to the output channels.

        * device (system default), (list/int):
            list of input and output devices;

        * inChannels ([1]), (list/int):
            list of device's input channel used for recording;

        * outChannels ([1]), (list/int):
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

    def __init__(self, excitation=None, outputAmplification=0, *args, **kwargs):
        if excitation is None:
            self._excitation = None
            super().__init__(*args, **kwargs)
        else:
            self.excitation = excitation
            if 'freqMin' in kwargs:
                kwargs.pop('freqMin')
            if 'freqMax' in kwargs:
                kwargs.pop('freqMax')
            super().__init__(*args,
                             samplingRate=excitation.samplingRate,
                             freqMin=excitation.freqMin,
                             freqMax=excitation.freqMax,
                             fftDegree=excitation.fftDegree,
                             timeLength=excitation.timeLength,
                             lengthDomain=excitation.lengthDomain,
                             numSamples=excitation.numSamples,
                             **kwargs)
            self.outChannel = excitation.channels
        self.outputAmplification = outputAmplification
        return

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                # PlayRecMeasure properties
                f'excitation={self.excitation!r}, '
                f'outputAmplification={self.outputAmplification!r}, '
                # Measurement properties
                f'device={self.device!r}, '
                f'inChannels={self.inChannels!r}, '
                f'outChannels={self.outChannels!r}, '
                f'blocking={self.blocking!r}, '
                # PyTTaObj properties
                f'samplingRate={self.samplingRate!r}, '
                f'freqMin={self.freqMin!r}, '
                f'freqMax={self.freqMax!r}, '
                f'comment={self.comment!r})')

# PlayRec Methods
    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Outputs a signalObj with the recording content
        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(framenline, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        timeStamp = time.ctime(time.time())
        recording = sd.playrec(self.excitation.timeSignal*
                               self.outputLinearGain,
                               samplerate=self.samplingRate,
                               input_mapping=self.inChannels.mapping,
                               output_mapping=self.outChannels.mapping,
                               device=self.device,
                               blocking=self.blocking,
                               latency='low',
                               dtype='float32')
        recording = np.squeeze(recording)
        recording = SignalObj(signalArray=recording*self.inChannels.CFlist(),
                              domain='time',
                              samplingRate=self.samplingRate,
                              freqMin=self.freqMin,
                              freqMax=self.freqMax)
        recording.channels = self.inChannels
        recording.timeStamp = timeStamp
        recording.comment = 'SignalObj from a PlayRec measurement'
        recording.creation_name = creation_name
        _print_max_level(self.excitation, kind='output',
                         gain=self.outputLinearGain,
                         mapping=self.outChannels.mapping)
        _print_max_level(recording, kind='input')
        return recording

    def _to_dict(self):
        sup = super()._to_dict()
        sup['excitationAddress'] = self.excitation._to_dict()
        return sup

    def pytta_save(self, dirname=time.ctime(time.time())):
        dic = self._to_dict()
        name = dirname + '.pytta'
        with zipfile.ZipFile(name, 'w') as zdir:
            excit = self.excitation.pytta_save('excitation')
            dic['excitationAddress'] = excit
            zdir.write(excit)
            os.remove(excit)
            with open('PlayRecMeasure.json', 'w') as f:
                json.dump(dic, f, indent=4)
            zdir.write('PlayRecMeasure.json')
            os.remove('PlayRecMeasure.json')
        return name

    def _h5_save(self, h5group, setClass=True):
        """
        Saves itself inside a hdf5 group from an already openned file via
        pytta._h5_save(...). Use setClass=True if the attribute 'class' must be
        seted to PlayRecMeasure.

        >>> PlayRecMeasure._h5_save(h5group, setClass=True)

        """
        if setClass is True:
            h5group.attrs['class'] = 'PlayRecMeasure'
        self.excitation._h5_save(h5group.create_group('excitation'))
        h5group.attrs['outputAmplification'] = self.outputAmplification
        super()._h5_save(h5group)
        pass

# PlayRec Properties
    @property
    def excitation(self):
        return self._excitation

    @excitation.setter
    def excitation(self, newSignalObj):
        self._excitation = newSignalObj
        return

    @property
    def outputAmplification(self):
        return self._outputAmplification

    @outputAmplification.setter
    def outputAmplification(self, newOutputGain):
        self._outputAmplification = newOutputGain
        self.outputLinearGain = 10**(self._outputAmplification/20)
        return

#    @property
#    def samplingRate(self):
#        return self.excitation._samplingRate
#
#    @property
#    def fftDegree(self):
#        return self.excitation._fftDegree
#
#    @property
#    def timeLength(self):
#        return self.excitation._timeLength
#
#    @property
#    def numSamples(self):
#        return self.excitation._numSamples
#
#    @property
#    def freqMin(self):
#        return self.excitation._freqMin
#
#    @property
#    def freqMax(self):
#        return self.excitation._freqMax


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

        * inChannels ([1]), (list | int):
            list of device's input channel used for recording;

        * outChannels ([1]), (list | int):
            list of device's output channel used for playing or reproducing
            a signalObj;

        * samplingRate (44100), (int):
            signal's sampling rate;

        * lengthDomain ('time'), (str):
            signal's length domain. May be 'time' or 'samples';

        * timeLength (seconds), (float):
            signal's time length in seconds for lengthDomain = 'time';

        * fftDegree (fftDegree), (float):
            2**fftDegree signal's number of samples for
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
                 # Coordinate and orientation management being done trough
                 # ChannelObj at in/out ChannelsList
                 #  coordinates={'points': [],
                 #               'reference': 'south-west-floor corner',
                 #               'unit': 'm'},
                 method='linear', winType=None, winSize=None,
                 overlap=None, regularization=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.coordinates = coordinates
        self.method = method
        self.winType = winType
        self.winSize = winSize
        self.overlap = overlap
        self.regularization = regularization
        return

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                # FRFMeasure properties
                f'method={self.method!r}, '
                f'winType={self.winType!r}, '
                f'winSize={self.winSize!r}, '
                f'overlap={self.overlap!r}, '
                # PlayRecMeasure properties
                f'excitation={self.excitation!r}, '
                f'outputAmplification={self.outputAmplification!r}, '
                # Measurement properties
                f'device={self.device!r}, '
                f'inChannels={self.inChannels!r}, '
                f'outChannels={self.outChannels!r}, '
                f'blocking={self.blocking!r}, '
                # PyTTaObj properties
                f'samplingRate={self.samplingRate!r}, '
                f'freqMin={self.freqMin!r}, '
                f'freqMax={self.freqMax!r}, '
                f'comment={self.comment!r})')

    def pytta_save(self, dirname=time.ctime(time.time())):
        dic = self._to_dict()
        name = dirname + '.pytta'
        with zipfile.ZipFile(name, 'w') as zdir:
            excit = self.excitation.pytta_save('excitation')
            dic['excitationAddress'] = excit
            zdir.write(excit)
            os.remove(excit)
            with open('FRFMeasure.json', 'w') as f:
                json.dump(dic, f, indent=4)
            zdir.write('FRFMeasure.json')
            os.remove('FRFMeasure.json')
        return name

    def _h5_save(self, h5group, setClass=True):
        """
        Saves itself inside a hdf5 group from an already openned file via
        pytta._h5_save(...). Use setClass=True if the attribute 'class' must be
        seted to FRFMeasure.

        >>> FRFMeasure._h5_save(h5group, setClass=True)

        """
        if setClass is True:
            h5group.attrs['class'] = 'FRFMeasure'
        h5group.attrs['method'] = _h5.none_parser(self.method)
        h5group.attrs['winType'] = _h5.none_parser(self.winType)
        h5group.attrs['winSize'] = _h5.none_parser(self.winSize)
        h5group.attrs['overlap'] = _h5.none_parser(self.overlap)
        super()._h5_save(h5group, setClass=False)
        pass

    def run(self):
        """
        Starts reproducing the excitation signal and recording at the same time
        Outputs the transferfunction ImpulsiveResponse
        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        recording = super().run()
        transferfunction = ImpulsiveResponse(self.excitation,
                                             recording,
                                             self.method,
                                             self.winType,
                                             self.winSize,
                                             self.overlap,
                                             self.regularization)
        transferfunction.timeStamp = recording.timeStamp
        transferfunction.creation_name = creation_name
        return transferfunction


# Sub functions
def _print_max_level(sigObj, kind, gain=1, mapping=None):
    for chIndex in range(sigObj.numChannels):
        chNum = sigObj.channels.mapping[chIndex]
        if mapping is not None:
            chNumMap = mapping[chIndex]
        else:
            chNumMap = chNum
        # Calculating the final level with a linear gain applied
        linearRmsAmplitude = 10**(sigObj.max_level()[chIndex]/20)
        finalLevel = 20*np.log10(linearRmsAmplitude*gain)
        print('max {} level (excitation) on channel [{}]: '
                .format(kind, chNumMap) +
                '{:.2f} {} - ref.: {} [{}]'
                .format(finalLevel,
                        sigObj.channels[chNum].dBName,
                        sigObj.channels[chNum].dBRef,
                        sigObj.channels[chNum].unit))
        if finalLevel >= 0:
            print('\x1b[0;30;43mATTENTION! CLIPPING OCCURRED\x1b[0m')
    return
