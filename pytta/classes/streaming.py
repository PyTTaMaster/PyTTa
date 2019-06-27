# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
from typing import Optional, List
from . import _base
from .signal import SignalObj


# Streaming class
class Streaming(_base.PyTTaObj):
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
                 inChannels: Optional[List[_base.ChannelObj]] = None,
                 outChannels: Optional[List[_base.ChannelObj]] = None,
                 duration: Optional[float] = None,
                 excitationData: Optional[np.ndarray] = None,
                 callback: Optional[callable] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_channels(inChannels, outChannels, excitationData)
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
        self.callback = callback
        self._call_for_stream(self.callback)
        return

    def _set_channels(self, inputs, outputs, data):
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
        self.inData = np.append(self.inData[:]*self.inChannels.CFlist(),
                                Idata, axis=0)
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
        self.inData = np.append(self.inData[:]*self.inChannels.CFlist(),
                                Idata, axis=0)
        try:
            Odata[:, :] = self.outData[self.kn:self.kn+frames, :]
            self.kn = self.kn + frames
        except ValueError:
            olen = len(self.outData[self.kn:self.kn+frames])
            Odata[:olen, :] = self.outData[self.kn:, :]
            Odata.fill(0)
            self.__timeout()
        return

    def __timeout(self):
        self.stop()
        self._call_for_stream(self.callback)
        self.kn = 0
        if self.inData is not None:
            self.inData = self.inData[1:, :]
        return

    def getSignal(self):
        signal = SignalObj(self.inData, 'time', self.samplingRate)
        return signal

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

    def calib_pressure(self, chIndex, refSignalObj,
                       refPrms=1.00, refFreq=1000):
        """
        .. method:: calibPressure(chIndex, refSignalObj, refPrms, refFreq):
            use informed SignalObj, with a calibration acoustic pressure
            signal, and the reference RMS acoustic pressure to calculate the
            Correction Factor and apply to every incoming audio on specified
            channel.

            >>> Streaming.calibPressure(chIndex,refSignalObj,refPrms,refFreq)

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

        if chIndex in range(len(self.inChannels)):
            self.inChannels[chIndex].calib_press(
                    refSignalObj, refPrms, refFreq)
            self.inChannels[chIndex].calibCheck = True
        else:
            raise IndexError('chIndex greater than channels number')
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
