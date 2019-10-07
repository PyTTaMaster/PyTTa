# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
from multiprocessing import Queue, Process, Event
from queue import Empty, Full
from typing import Optional, List, Callable, Union, Generic
from pytta.classes import _base
from pytta.classes.signal import SignalObj
from pytta.classes.measurement import Measurement, RecMeasure


# Recording obj class
class Recorder(object):
    """
    Recorder:
    ----------

        Provides a recorder object that executes, in a parallel process some function
        with the incoming data.
    """
    def __init__(self, msmnt: Measurement,
                 datatype: str='float32',
                 blocksize: int=32):
        """

        :param msmnt: PyTTa Measurement-like object.
        :type msmnt: pytta.RecMeasure
        :param datatype: string with the data type name
        :type datatype: str
        :param blocksize: number of frames reads on each call of the stream callback
        :type blocksize: int
        """
        self.samplingRate = msmnt.samplingRate  # registers samples per second
        self.numSamples = msmnt.numSamples  # registers total amount of samples recorded
        self.numChannels = msmnt.numInChannels  # register number of channels to be recorded
        self.device = msmnt.device  # registers device identification
        self.dataType = datatype  # registers data type
        self.blocksize = blocksize  # registers blocksize
        self.switch = Event()  # instantiates a multiprocessing Event object
        self.monitor = Event()
        """
        Essentially, the Event object is a boolean state. It can be
        `.set()` : Internally defines it to be True;
        `.clear()` : Internally defines it to be False;
        `.is_set()` : Check if it is True (only after call to `.set()`)
        """
        self.last_status = None  # will register last status passed by stream
        self.recCount = int()  # registers number of frames recorded
        self.recQueue = Queue(self.numSamples//16)  # instantiates a multiprocessing Queue
        """
        A Queue is First In First Out (FIFO) container object. Data can be stored in it
        and be retrieved in the same order as it has been put. It can
        `.put()` : Add data to Queue
        `.put_nowait()` : Add data to Queue without waiting for memlocks
        `.get()` : Retrieve data from Queue
        `.get_nowait()` : Retrieves data from Queue without waiting for memlocks   
        """
        self.recData = np.empty((self.numSamples,
                                 self.numChannels),
                                dtype=self.dataType)  # allocates memory for data as numpy array
        return

    def __enter__(self):
        """
        Provides context functionality, e.g.

            >>> with Recorder(msmnt) as rec:  <-- called here
            ...     rec.set_monitoring(somefunc)
            ...     rec.run()

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Provides context functionality, e.g.

            >>> with Recorder(msmnt) as rec:
            ...     rec.set_monitoring(somefunc)
            ...     rec.run()
            ...                                 <-- called here
            >>>
        """
        if exc_tb:
            raise exc_val
        else:
            return

    def set_monitoring(self, func: Callable):
        """
        Set up the function used as monitor. It must have the following declaration:

            def monitor_callback(data: np.ndarray,
                                 frames: int,
                                 status: sd.CallbackFlags)

        It will be called from within a parallel process that the Recorder starts and
        terminates during it's .run() call.

        :param func:
        :type func: Callable
        """
        if func is False:
            self.monitor.clear()
            return
        elif isinstance(func, Callable):
            self.monitor_callback = func
            self.monitor.set()
        else:
            raise ValueError("The monitoring argument must be a callable:",
                             "a function or a method.")
        return

    def stream_callback(self, indata: np.ndarray, frames: int,
                        times: type, status: sd.CallbackFlags):
        """
        This method will be called from the stream, as stated on sounddevice's documentation.
        """
        try:
            self.recData[self.recCount:self.recCount + frames, :] = indata[:, :]
        except IndexError:
            self.recData[self.recCount:, :] = indata[:self.numSamples-self.recCount, :]
        if self.monitor:
            self.recQueue.put_nowait([indata, frames, status])
        self.recCount += frames
        if self.recCount >= self.numSamples:
            raise sd.CallbackStop
        return

    def parallel_loop(self):
        """
        This function is the parallel process' loop, that is responsible for getting
        the data from queue and passing it to the monitor function, if there is one.

        :return:
        :rtype:
        """
        while not self.switch.is_set():  # this loop waits for the switch to be turned
            if self.switch.is_set():     # on before continuing
                break
            else:
                continue
        while self.switch.is_set():  # this loop tries to read from the queue, after
            try:                     # call to switch.set()
                data, frames, status = self.recQueue.get_nowait()  # get from queue
                if status:   # check any status
                    self.last_status = status
                    print(status)  # prints status to stdout, for checking
                self.monitor_callback(data, frames, status)  # calls for monitoring function
            except Empty:  # if queue has no data, checks if callback is stopped
                if self.last_status is sd.CallbackStop:
                    break
                else:
                    continue
        return

    def run(self):
        """
        Instantiates a sounddevice.InputStream and calls for a parallel process
        if any monitoring is set up.
        Then turn on the switch Event, and starts the stream.
        Waits for it to finish, unset the event
        And terminates the process

        :return:
        :rtype:
        """
        with sd.InputStream(samplerate=self.samplingRate,
                            blocksize=self.blocksize,
                            device=self.device,
                            channels=self.numChannels,
                            dtype=self.dataType,
                            latency='low',
                            callback=self.stream_callback) as stream:
            if self.monitor:
                Parallel = Process(target=self.parallel_loop)
                Parallel.start()
            self.switch.set()
            stream.start()
            while stream.active:
                if stream.stopped:
                    break
                else:
                    continue
            stream.stop()  # just to be sure...
            self.switch.clear()
            if self.monitor:
                Parallel.terminate()
            stream.close()
        return


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

    def calib_pressure(self, chIndex, refPrms=1.00, refFreq=1000):
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
                channel number for calibration;

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
                                  device=self.device).run()
        if chIndex-1 in self.inChannels.mapping():
            self.inChannels[chIndex-1].calib_press(refSignalObj, refPrms, refFreq)
            self.inChannels[chIndex-1].calibCheck = True
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
