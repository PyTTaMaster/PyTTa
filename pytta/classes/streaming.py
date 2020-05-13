# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
import multiprocessing as mp
from queue import Empty, Queue
from threading import Event
from typing import Optional, Callable, List, Type, Union
from pytta import default, utils
from pytta.classes._base import PyTTaObj
from pytta.classes.signal import SignalObj
from pytta.classes.measurement import Measurement, RecMeasure


class Monitor(object):
    """PyTTa default Monitor base class."""

    def __init__(self, numsamples: int,
                 samplingrate: int = default.samplingRate,
                 numchannels: List[int] = [len(default.inChannel),
                                           len(default.outChannel)],
                 datatype: str = 'float32'):
        self.samplingRate = samplingrate
        self.numChannels = numchannels
        self.numSamples = numsamples
        self.dtype = datatype
        return

    def setup(self):
        """
        Start up widgets, threads, anything that will be used during audio processing
        """
        self.inData = np.empty((self.numSamples, self.numChannels[0]), dtype=self.dtype)
        self.outData = np.empty((self.numSamples, self.numChannels[1]), dtype=self.dtype)
        self.red = utils.ColorStr("white", "red")
        self.green = utils.ColorStr("white", "green")
        self.yellow = utils.ColorStr("black", "yellow")
        return

    def reset(self):
        self.counter = int()
        return

    def callback(self, frames: int,
                 indata: np.ndarray,
                 outdata: Optional[np.ndarray] = None):
        """
        The audio processing itself, will be called for every chunk of data taken from the queue
        """
        if self.inData.shape[0] >= self.samplingRate//8:
            indB = utils.arr2dB(self.inData)
            outdB = utils.arr2dB(self.outData)
            if indB >= -3:
                indBstr = self.red(f'{indB}')
            elif indB >= -10:
                indBstr = self.yellow(f'{indB}')
            else:
                indBstr = self.green(f'{indB}')
            if outdB >= -3:
                outdBstr = self.red(f'{outdB}')
            elif outdB >= -10:
                outdBstr = self.yellow(f'{outdB}')
            else:
                outdBstr = self.green(f'{outdB}')
            print('input fast rms:', indBstr, 'output fast rms:', outdBstr, end='\r')
            self.reset()
        else:
            self.inData[self.counter:self.counter+frames] = indata[:]
            self.outData[self.counter:self.counter+frames] = outdata[:] if outdata is (not None) else 0
            self.counter += frames
        return

    def tear_down(self):
        """
        Finish any started object here, like GUI members, to allow the Monitor parallel process be joined
        """
        pass


# Streaming class
class Streaming(PyTTaObj):
    """    """

    def __init__(self,
                 IO: str,
                 msmnt: Measurement,
                 datatype: str = 'float32',
                 blocksize: int = 0,
                 duration: Optional[float] = None,
                 monitor: Optional[Monitor] = None,
                 *args, **kwargs):
        """
        Streaming:
        ---------

        Args:
            IO (str): DESCRIPTION.
            msmnt (Measurement): DESCRIPTION.
            datatype (str, optional): DESCRIPTION. Defaults to 'float32'.
            blocksize (int, optional): DESCRIPTION. Defaults to 0.
            duration (Optional[float], optional): DESCRIPTION. Defaults to 5.
            monitor (Optional[Monitor], optional): DESCRIPTION. Defaults to None.
            *args (TYPE): DESCRIPTION.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            None.

        """

        super().__init__(*args, **kwargs)
        self._IO = IO.upper()
        self._samplingRate = msmnt.samplingRate  # registers samples per second
        self._numSamples = msmnt.numSamples  # registers total amount of samples recorded
        self._dataType = datatype  # registers data type
        self._blockSize = blocksize  # registers blocksize
        if (type(duration) is float) or (type(duration) is int):
            self._durationInSamples = int(duration*msmnt.samplingRate)
        else:
            self._durationInSamples = self.numSamples
        self._duration = duration
        self._device = msmnt.device
        self._theEnd = False
        self.loopCheck = Event()  # controls looping state
        self.loopCheck.clear()  # ensure False
        self.monitorCheck = Event()  # monitoring state
        self.runningCheck = Event()  # measurement state
        self.runningCheck.clear()  # ensure False
        self.lastStatus = []  # will register last status passed by stream
        # self.statusCount = int(0)  # zero
        self.queue = Queue(self.numSamples//2)  # instantiates a multiprocessing Queue
        self.set_monitoring(monitor)
        self.set_io_properties(msmnt)
        return

    def __enter__(self):
        """
        Provide context functionality, the `with` keyword, e.g.

            >>> with Recorder(Measurement) as rec:  # <-- called here
            ...     rec.set_monitoring(Callable)
            ...     rec.run()
            ...
            >>>

        """
        return self

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: Type):
        """
        Provide context functionality, the `with` keyword, e.g.

            >>> with Streaming('play', Measurement) as strm:
            ...     strm.set_monitoring(Callable)
            ...     strm.run()
            ...                             # <-- called here
            >>>
        """
        if exc_tb:
            raise exc_val
        else:
            return

    def set_io_properties(self, msmnt):
        """
        Allocate memory for input and output of data, set counter.

        Args:
            msmnt (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        if 'I' in self.IO:
            self.inChannels = msmnt.inChannels
            self.recData = np.empty((self.durationInSamples, self.numInChannels),
                                    dtype=self.dataType)
        if 'O' in self.IO:
            self.outChannels = msmnt.outChannels
            self.playData = msmnt.excitation.timeSignal[:]
        self.dataCount = int(0)
        return

#    def play_data_adjust(self, playdata):
#        len = playdata.shape[0]
#        chn = playdata.shape[1]
#        bs = self.blockSize
#        nchunks = int(np.ceil(len / bs))
#        array = np.empty((nchunks, bs, chn), dtype='float32')
#        for c in range(chn):
#            for n in range(nchunks):
#                array[n, :, c] = playdata[n * bs:(n + 1) * bs, c]
#        return array
#
#    def rec_data_adjust(self, nsamples, nchannels):
#        bs = self.blockSize
#        nchunks = int(np.ceil(nsamples / bs))
#        array = np.empty((nchunks, bs, nchannels), dtype='float32')
#        return array
#

    def set_monitoring(self,
                       monitor: Monitor = None):
        """
        Set up the class used as monitor. It must have the following methods with these names.

            def setup(None) -> None:

                _Call any function and other object configuration needed for the monitoring_
                return


            def callback(indata: np.ndarray, outdata: np.ndarray,
                         frames: int, status: sd.CallbackFlags) -> None:

                _Process the data gathered from the stream_
                return

        It will be called from within a parallel process that the Recorder starts and
        terminates during it's .run() call.

        :param monitor: Object or class that will be used to monitor the stream data flow.
        :type monitor: object

        """
        if monitor is None:
            self.monitorCheck.clear()  # ensure False
        else:
            self.monitorCheck.set()  # set to True
        self.monitor = monitor
        return

    def monitoring1(self):
        """
        Monitor loop.

        This is the place where the parallel processing occurs, the income of
        data and the calling for the callback function. Tear down after loop
        breaks.

        Returns:
            None.

        """
        self.monitor.setup()    # calls the Monitor function to set up itself
        while not self.runningCheck.is_set():  # Waits untill stream start
            continue
        while self.runningCheck.is_set():
            try:
                streamCallbackData = self.queue.get_nowait()  # get from queue
                if streamCallbackData[-1]:  # check any status
                    self.lastStatus = streamCallbackData[-1]  # saves the last
                    self.statusCount += 1  # counts one more
                self.monitor.callback(*streamCallbackData)  # calls for monitoring function
            except Empty:  # if queue has no data
                continue
        self.monitor.tear_down()
        return

    def streaming(self,
                  streamtype: sd.Stream,
                  streamcallback: Callable):
        """
        Stream loop.

        Attempt to run the streaming in parallel process and monitoring on main.
        """
        self.stream = streamtype(samplerate=self.samplingRate,
                                 blocksize=self.blockSize,
                                 device=self.device,
                                 channels=self.numChannels,
                                 dtype=self.dataType,
                                 latency='low',
                                 dither_off=True,
                                 callback=streamcallback)
        self.runningCheck.set()
        self.stream.start()
        sd.sleep(int(self.duration*1000)+10)
        self.stream.stop()
        self.stream.close()
        self.runningCheck.clear()
        return

    def monitoring(self):
        while not self.runningCheck.is_set():  # Waits untill stream start
            continue
        while self.runningCheck.is_set():
            try:
                streamCallbackData = self.queue.get_nowait()  # get from queue
                streamStatus = streamCallbackData.pop()
                if streamStatus is not None:  # check any status
                    self.lastStatus = streamStatus  # saves the last
                    self.statusCount += 1  # counts one more
                self.monitor.callback(*streamCallbackData)  # calls for monitoring function
            except Empty:  # if queue has no data
                if self.runningCheck.is_set():
                    continue
                else:
                    break
        return

    def runner1(self, streamtype, streamcallback, numchannels):
        """
        Loop attempt 1. WORST CASE.

        Args:
            streamtype (TYPE): DESCRIPTION.
            streamcallback (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        self.stream = streamtype(samplerate=self.samplingRate,
                                 blocksize=self.blockSize,
                                 device=self.device,
                                 channels=numchannels,
                                 dtype=self.dataType,
                                 latency='low',
                                 dither_off=True,
                                 callback=streamcallback)
        if self.monitorCheck.is_set():
            self.monitor.setup()  # calls the Monitor function to set up itself
        self.stream.start()
        while self.stream.active:
            try:
                streamCallbackData = self.queue.get_nowait()
                streamStatus = streamCallbackData.pop(-1)
                if streamStatus is not None:
                    self.lastStatus.append(streamStatus)
                if self.monitorCheck.is_set():
                    self.monitor.callback(*streamCallbackData)
            except Empty:  # if queue has no data
                if self.stream.stopped:
                    break
                else:
                    continue
        self.stream.close()
        if self.monitorCheck.is_set():
            self.monitor.tear_down()
        return

    def runner2(self, streamtype: Type, streamcallback: Callable):
        """
        Loop attemp 2.

        This is the place where the parallel processing occurs, the income of
        data and the calling for the callback function. Tear down after loop
        breaks.

        Returns:
            None.

        """
        if self.monitorCheck.is_set():
            self.monitor.setup()    # calls the Monitor function to set up itself
        process = mp.Process(target=self.streaming, args=(streamtype, streamcallback))
        process.start()
        if self.monitorCheck.is_set():
            self.monitoring()
        process.join()
        process.close()
        if self.monitorCheck.is_set():
            self.monitor.tear_down()
        return

    def runner(self, StreamType: Type, stream_callback: Callable, numchannels: Union[List[int], int]):
        """
        Loop attempt 3.

        Instantiates a sounddevice.InputStream and calls for a parallel process
        if any monitoring is set up.
        Then turn on the monitorCheck Event, and starts the stream.
        Waits for it to finish, unset the event
        And terminates the process

        :return:
        :rtype:
        """
        if self.monitorCheck.is_set():
            self.monitor.setup()
            t = mp.Process(target=self.monitoring)
            t.start()
        self.runningCheck.set()
        with StreamType(samplerate=self.samplingRate,
                        blocksize=self.blockSize,
                        device=self.device,
                        channels=numchannels,
                        dtype=self.dataType,
                        latency='low',
                        dither_off=True,
                        callback=stream_callback):
            sd.sleep(int(self.duration * self.blockSize) + 10)
        self.runningCheck.clear()
        if self.monitorCheck.is_set():
            t.join()
            t.close()
            self.monitor.tear_down()
        return

    def input_callback(self, indata: np.ndarray, frames: int,
                       times: type, status: sd.CallbackFlags):
        """This method will be called from the stream, as stated on sounddevice's documentation."""
        writesLeft = self.recData.shape[0] - self.dataCount - 1
        framesWrite = writesLeft+1 if writesLeft < frames else frames
        self.recData[self.dataCount:framesWrite + self.dataCount, :] = indata[:framesWrite]
        if self.monitor:
            self.queue.put_nowait([indata[:], None, frames, status])
        self.dataCount += frames
        if self.dataCount >= self.durationInSamples:
            raise sd.CallbackStop
        return

    def output_callback(self, outdata: np.ndarray, frames: int,
                      times: type, status: sd.CallbackFlags):
        """This method will be called from the stream, as stated on sounddevice's documentation."""
        readsLeft = self.playData.shape[0] - self.dataCount - 1
        framesRead = readsLeft+1 if readsLeft < frames else frames
        outdata[:framesRead] = self.playData[self.dataCount:framesRead + self.dataCount, :]
        outdata[framesRead:].fill(0.)
        if self.monitor:
            self.queue.put_nowait([None, outdata[:], frames, status])
        self.dataCount += frames
        if self.dataCount >= self.durationInSamples:
            raise sd.CallbackStop
        return

    def stream_callback(self, indata, outdata, frames, time, status):
        """This method will be called from the stream, as stated on sounddevice's documentation."""
        writesLeft = self.recData.shape[0] - self.dataCount - 1
        framesWrite = writesLeft+1 if writesLeft < frames else frames
        self.recData[self.dataCount:framesWrite + self.dataCount, :] = indata[:framesWrite]
        readsLeft = self.playData.shape[0] - self.dataCount - 1
        framesRead = readsLeft+1 if readsLeft < frames else frames
        outdata[:framesRead] = self.playData[self.dataCount:framesRead + self.dataCount, :]
        outdata[framesRead:].fill(0)
        if self.monitorCheck.is_set():
            self.queue.put_nowait([indata.copy(), outdata.copy(), frames, status])
        self.dataCount += frames
        if self.dataCount >= self.durationInSamples:
            raise sd.CallbackStop
        return

    def play(self):
        self.runner(sd.OutputStream, self.output_callback, self.numOutChannels)
        self.dataCount = int()
        return

    def record(self):
        self.runner(sd.InputStream, self.input_callback, self.numInChannels)
        self.dataCount = int()
        return self.recData

    def playrec(self):
        self.runner(sd.Stream, self.stream_callback, self.numChannels)
        self.dataCount = int()
        return self.recData

    @property
    def IO(self):
        return self._IO

    @property
    def device(self):
        return self._device

    @property
    def blockSize(self):
        return self._blockSize

    @property
    def dataType(self):
        return self._dataType

    @property
    def duration(self):
        return self._durationInSamples/self.samplingRate

    @property
    def durationInSamples(self):
        return self._durationInSamples

    @property
    def numInChannels(self):
        return len(self.inChannels)

    @property
    def numOutChannels(self):
        return len(self.outChannels)

    @property
    def theEnd(self):
        return self._theEnd

    @property
    def numChannels(self):
        if self.IO == 'I':
            return self.numInChannels
        elif self.IO == 'O':
            return self.numOutChannels
        elif self.IO == 'IO':
            return self.numInChannels, self.numOutChannels


# Recording obj class
class Recorder(Streaming):
    """
    Recorder:
    ----------

        Provides a recorder object that executes, in a parallel process some function
        with the incoming data.
    """
    def __init__(self, msmnt: Measurement,
                 datatype: str='float32',
                 blocksize: int=0,
                 duration: Optional[float] = 5,
                 *args, **kwargs):
        """

        :param msmnt: PyTTa Measurement-like object.
        :type msmnt: pytta.RecMeasure
        :param datatype: string with the data type name
        :type datatype: str
        :param blocksize: number of frames reads on each call of the stream callback
        :type blocksize: int
        """
        super().__init__('I', msmnt, datatype, blocksize, duration, *args, **kwargs)
        return

    def stream_callback(self, indata: np.ndarray, frames: int,
                        times: type, status: sd.CallbackFlags):
        """
        This method will be called from the stream, as stated on sounddevice's documentation.
        """
        self.recData[self.dataCount:frames + self.dataCount, :] = indata[:]
        if self.monitor:
            self.queue.put_nowait((indata[:], None, frames, status))
        self.dataCount += frames
        if self.dataCount >= self.durationInSamples:
            raise sd.CallbackStop
        return

    def retrieve(self):
        arr = self.recData.reshape((self.numSamples, self.numInChannels))
        assert arr.ndim == 2
        signal = SignalObj(arr, 'time', self.samplingRate,
                           freqMin=20, freqMax=20e3)
        return signal

    def run(self):
        self.runner(sd.InputStream, self.stream_callback)
        return

# Playback obj class
class Player(Streaming):
    """
    Recorder:
    ----------

        Provides a recorder object that executes, in a parallel process some function
        with the incoming data.
    """
    def __init__(self, msmnt: Measurement,
                 datatype: str='float32',
                 blocksize: int=0,
                 *args, **kwargs):
        """

        :param msmnt: PyTTa Measurement-like object.
        :type msmnt: pytta.RecMeasure
        :param datatype: string with the data type name
        :type datatype: str
        :param blocksize: number of frames reads on each call of the stream callback
        :type blocksize: int
        """
        super().__init__('O', msmnt, datatype, blocksize, *args, **kwargs)
        return

    def stream_callback(self, outdata: np.ndarray, frames: int,
                        times: type, status: sd.CallbackFlags):
        """
        This method will be called from the stream, as stated on sounddevice's documentation.
        """
        outdata[:] = self.playData[self.dataCount:frames + self.dataCount, :]
        if self.monitor:
            self.queue.put_nowait((None, outdata[:], frames, status))
        self.dataCount += frames
        if self.dataCount >= self.durationInSamples:
            raise sd.CallbackStop
        return

    def run(self):
        """
        Instantiates a sounddevice.OutputStream and calls for a parallel process
        if any monitoring is set up.
        Then turn on the monitorCheck Event, and starts the stream.
        Waits for it to finish, unset the event
        And terminates the process

        :return:
        :rtype:
        """
        self.runner(sd.OutputStream, self.stream_callback)
        return


class PlaybackRecorder(Streaming):
    """
    ...
    """
    def __init__(self, msmnt: Measurement,
                 datatype: str = 'float32',
                 blocksize: int = 0):
        super().__init__('IO', msmnt, datatype, blocksize)
        return

    def stream_callback(self, indata, outdata, frames, time, status):
        try:
            outdata[:] = self.playData[self.dataCount:frames + self.dataCount, :]
            self.recData[self.dataCount:frames + self.dataCount, :] = indata[:]
            if self.monitorCheck.is_set():
                self.queue.put_nowait((self.recData[self.dataCount:frames + self.dataCount, :],
                                       self.playData[self.dataCount:frames + self.dataCount, :],
                                       frames,
                                       status))
            self.dataCount += frames
        except ValueError:
            raise sd.CallbackStop
        except Exception as e:
            print(type(e), e, '\n', 'Last Callback Status:', status)
            raise sd.CallbackAbort
        return

    def retrieve(self):
        signal = SignalObj(self.recData, 'time', self.samplingRate,
                           freqMin=20, freqMax=20e3)
        return signal

    def run(self):
        self.runner(sd.Stream, self.stream_callback)
        return

