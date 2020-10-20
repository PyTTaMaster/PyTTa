# -*- coding: utf-8 -*-
"""
Provide real time audio playback and recording, with special classes to
concurrently read input audio.

"""

import numpy as np
import sounddevice as sd
#import multiprocessing as mp
from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional, Callable, List, Type, Union
from pytta import default, utils
from pytta.classes._base import PyTTaObj, ChannelsList
from pytta.classes.signal import SignalObj

# TO DO: format docs

class Monitor(object):
    """PyTTa default Monitor base class."""

    def __init__(self, numsamples: int,
                 samplingrate: int = default.samplingRate,
                 numchannels: List[int] = [len(default.inChannel),
                                           len(default.outChannel)],
                 datatype: str = 'float32'):
        """
        Default Monitor class.

        Subclasses must override `setup`, `callback` and `tear_down` methods.

        Parameters
        ----------
        numsamples : int
            DESCRIPTION.
        samplingrate : int, optional
            DESCRIPTION. The default is default.samplingRate.
        numchannels : List[int], optional
            DESCRIPTION.
            The default is [len(default.inChannel), len(default.outChannel)].
        datatype : str, optional
            DESCRIPTION. The default is 'float32'.

        Returns
        -------
        None.

        """
        self.samplingRate = samplingrate
        self.numChannels = numchannels
        self.numSamples = numsamples
        self.loopDuration = self.numSamples / self.samplingRate
        self.dtype = datatype
        return

    def setup(self):
        """Start up widgets, threads, anything that will be used during audio processing."""
        self.inData = np.empty((self.numSamples, self.numChannels[0]), dtype=self.dtype)
        self.outData = np.empty((self.numSamples, self.numChannels[1]), dtype=self.dtype)
        self.red = utils.ColorStr("white", "red")
        self.green = utils.ColorStr("white", "green")
        self.yellow = utils.ColorStr("black", "yellow")
        self.reset()
        #print('\r\tinput: 00.00 dB\toutput: 00.00 dB\t', end='\r')
        return

    def reset(self):
        """Reset write counter."""
        self.counter = int()
        return


    def callback(self, frames: int,
                 indata: Optional[np.ndarray] = None,
                 outdata: Optional[np.ndarray] = None):
        """
        The audio processing / exhibiting function

        This will be called for every chunk of data taken from the queue.

        Parameters
        ----------
        frames : int
            Number of audio samples in each channel.
        indata : np.ndarray
            The recorded audio data, frames x numchannels.
        outdata : Optional[np.ndarray], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if self.counter >= self.samplingRate//8:
            indB = utils.arr2dB(self.inData)
            outdB = utils.arr2dB(self.outData)
            if indB >= -3:
                indBstr = self.red(f'{indB: ^8.1f}')
            elif indB >= -10 and indB < -3:
                indBstr = self.yellow(f'{indB: ^8.1f}')
            else:
                indBstr = self.green(f'{indB: ^8.1f}')
            if outdB >= -3:
                outdBstr = self.red(f'{outdB: ^8.1f}')
            elif outdB >= -10 and outdB < -3:
                outdBstr = self.yellow(f'{outdB: ^8.1f}')
            else:
                outdBstr = self.green(f'{outdB: ^8.1f}')
            print(f'\r\tinput: {indBstr} dB\toutput: {outdBstr} dB\t', end='\r')
            self.reset()
        else:
            writeIn = self.inData.shape[0] - self.counter \
                if self.counter + frames > self.inData.shape[0] \
                else frames
            writeOut = self.outData.shape[0] - self.counter \
                if self.counter + frames > self.outData.shape[0] \
                else frames
            self.inData[self.counter:self.counter + writeIn] = indata[:writeIn] \
                if indata is not None else 0
            self.outData[self.counter:self.counter + writeOut] = outdata[:writeOut] \
                if outdata is not None else 0
            self.counter += frames
        return

    def tear_down(self):
        """Finish any started object here, like GUI members, to allow the Monitor parallel process be joined."""
        print()
        return



class Streamer(object):
    """Analog to digital and digital to analog device abstraction class."""

    def __init__(self, idx, blocksize, samplingRate = None):
        self._dict = sd.query_devices(idx)
        self._bs = blocksize
        self._samplingRate = (self._dict['default_samplerate']
                              if samplingRate is None
                              else samplingRate)
        self._inChannels = ChannelsList([ch + 1 for ch in range(self.numInChannels)])
        self._outChannels = ChannelsList([ch + 1 for ch in range(self.numOutChannels)])
        return

    @property
    def name(self):
        """Device name."""
        return self._dict['name']

    @property
    def inChannels(self):
        """List of input channels."""
        return self._inChannels

    @property
    def numInChannels(self):
        """Amount of input channels."""
        return self._dict['max_input_channels']

    @property
    def outChannels(self):
        """List of output channels."""
        return self._outChannels

    @property
    def numOutChannels(self):
        """Amount of output channels."""
        return self._dict['max_output_channels']





# Streaming class
class Streaming(PyTTaObj):
    """Stream control."""

    def __init__(self,
                 IO: str,
                 samplingRate: int,
                 device: int,
                 inChannels: ChannelsList,
                 outChannels: ChannelsList,
                 datatype: str = 'float32',
                 blocksize: int = 0,
                 excitation: Optional[SignalObj] = None,
                 duration: Optional[float] = None,
                 numSamples: Optional[int] = None,
                 monitor: Optional[Monitor] = None,
                 *args, **kwargs):
        """
        Manage input and output of audio.

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

        self._samplingRate = samplingRate  # registers samples per second
        self._dataType = datatype  # registers data type
        self._blockSize = blocksize  # registers blocksize
        self._device = device

        if (type(duration) is float) or (type(duration) is int):
            self._durationInSamples = int(np.ceil(duration*self.samplingRate))
        else:
            self._durationInSamples = numSamples
        self._duration = self.durationInSamples / self.samplingRate

        self.isFinished = Event()  # block untill finished
        self.hasMonitor = Event()  # prevent threading if no monitor
        self.isRunning = Event()   # stream and monitor synchronization

        self.statuses = []  # will register statuses passed by stream
        if 'I' in self.IO:
            self.inChannels = inChannels
            self.recData = np.empty((self.durationInSamples, self.numInChannels),
                                    dtype=self.dataType)
        if 'O' in self.IO:
            self.outChannels = outChannels
            self.playData = excitation.timeSignal[:]
        self.dataCount = int(0)
        self.dulldata = np.zeros((self.durationInSamples, 1), dtype=self.dataType)
        self.set_monitoring(monitor)
        return

    def __enter__(self):
        """
        Provide context functionality, the `with` keyword, e.g.

            >>> with Streaming(*args, **kwargs) as strm:  # <-- called here
            ...     strm.playrec()
            ...
            >>>

        """
        return self

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: Type):
        """
        Provide context functionality, the `with` keyword, e.g.

            >>> with Streaming(*args, **kwargs) as strm:
            ...     strm.playrec()
            ...                             # <-- called here
            >>>
        """
        if exc_tb:
            raise exc_val
        else:
            return

    def set_io_properties(self, io: str, channels: ChannelsList):
        """
        Allocate memory for input and output of data, set counter.

        Args:
            msmnt (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        return

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
            self.hasMonitor.clear()  # ensure False
        else:
            self.hasMonitor.set()  # set to True
            self.queue = Queue()  # data queueing for monitor
        self.monitor = monitor
        return

    def _monitoring_thread_loop(self, monitor, running, queue, statuses):
        """
        Monitor loop.

        This is the place where the parallel processing occurs, the income of
        data and the calling for the callback function. Tear down after loop
        breaks.

        Returns:
            None.

        """
        monitor.setup()    # calls the Monitor function to set up itself
        running.wait()
        while running.is_set():
            sd.sleep(int(monitor.loopDuration * 1000))
            while True:
                try:
                    indata, outdata, frames, status = queue.get_nowait()  # get from queue
                    if status:  # check any status
                        statuses.append(status)  # saves the last
                    monitor.callback(frames, indata, outdata)  # calls for monitoring function
                except Empty:  # if queue has no data
                    break
        monitor.tear_down()
        return


    def runner(self, StreamType: Type, stream_callback: Callable, numchannels: Union[List[int], int]):
        """
        Do the work.

        Instantiates a sounddevice.*Stream and calls for a threading.Thread
        if any Monitor is set up.
        Then turn on the monitorCheck Event, and starts the stream.
        Waits for it to finish, unset the event
        And terminates the process

        :return:
        :rtype:
        """
        self.isFinished.clear()
        if self.hasMonitor.is_set():
            t = Thread(target=self._monitoring_thread_loop,
                       args=(self.monitor, self.isRunning,
                             self.queue, self.statuses))
            t.start()
        with StreamType(samplerate=self.samplingRate,           # frames per second
                        blocksize=self.blockSize,               # frames per call
                        device=self.device,                     # I/O devices
                        channels=numchannels,                   # number of channels FIXME: mapping
                        dtype=self.dataType,                    # type of sample data
                        latency='low',                          # request lowest possible latency
                        dither_off=True,                        # disable PortAudio dithering
                        clip_off=True,                          # disable PortAudio clipping
                        callback=stream_callback,               # callback for each type
                        finished_callback=self._end_of_stream): # called after Abort or Stop stream
            self.isRunning.set()
            self.isFinished.wait()
        self.isRunning.clear()
        if self.hasMonitor.is_set():
            t.join()
        return

    def _register_input_data(self, cbInput, frames):
        writesLeft = self.recData.shape[0] - self.dataCount
        framesWrite = writesLeft if writesLeft <= frames else frames
        self.recData[self.dataCount:framesWrite + self.dataCount, :] = cbInput[:framesWrite, [m-1 for m in self.inChannels.mapping]]
        return

    def _register_output_data(self, cbOutput, frames):
        readsLeft = self.playData.shape[0] - self.dataCount
        framesRead = readsLeft if readsLeft <= frames else frames
        cbOutput[:framesRead, [m-1 for m in self.outChannels.mapping]] = self.playData[self.dataCount:framesRead + self.dataCount, :]
        cbOutput[framesRead:].fill(0.)
        return

    def _end_of_callback(self, indata, outdata, frames, status):
        if self.hasMonitor.is_set():
            self.queue.put_nowait([indata.copy(),
                                   outdata.copy(),
                                   frames, status])
        self.dataCount += frames
        if self.dataCount >= self.durationInSamples:
            raise sd.CallbackStop
        return

    def _end_of_stream(self):
        self.isFinished.set()
        return

    def input_callback(self, indata: np.ndarray, frames: int,
                       times: type, status: sd.CallbackFlags):
        """This method will be called from the stream, as stated on sounddevice's documentation."""
        self._register_input_data(indata, frames)
        self._end_of_callback(indata, self.dulldata, frames, status)
        return

    def output_callback(self, outdata: np.ndarray, frames: int,
                      times: type, status: sd.CallbackFlags):
        """This method will be called from the stream, as stated on sounddevice's documentation."""
        self._register_output_data(outdata, frames)
        self._end_of_callback(self.dulldata, outdata, frames, status)
        return

    def stream_callback(self, indata: np.ndarray, outdata: np.ndarray,
                        frames: int, time: type, status: sd.CallbackFlags):
        """This method will be called from the stream, as stated on sounddevice's documentation."""
        self._register_output_data(outdata, frames)
        self._register_input_data(indata, frames)
        self._end_of_callback(indata, outdata, frames, status)
        return

    def play(self):
        self.runner(sd.OutputStream, self.output_callback, max(self.outChannels.mapping))
        self.dataCount = int()
        return

    def record(self):
        self.runner(sd.InputStream, self.input_callback, max(self.inChannels.mapping))
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
    def numChannels(self):
        if self.IO == 'I':
            return self.numInChannels
        elif self.IO == 'O':
            return self.numOutChannels
        elif self.IO == 'IO':
            return self.numInChannels, self.numOutChannels

