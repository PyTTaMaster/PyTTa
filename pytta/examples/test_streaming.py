# -*- coding: utf-8 -*-

from sys import stdout
import numpy as np
import sounddevice as sd
from pytta.classes.measurement import Measurement
from pytta.classes.streaming import Recorder, Player, PlaybackRecorder
from pytta.classes.signal import SignalObj
from typing import Callable

class DoomyDoode(object):
    integ = {
        'fast': 8,
        'slow': 1,
        'peak': 12.5
    }
    """
    Doomy Doode:
    -------------

        Dummy (duh!) class for simple default methods
        Can change the printing rate by changing the numSamples parameter.
        Changing the `integ` key is an option. 
    """
    def __init__(self, samplingrate, numchannels, blocksize):
        self.blocksize = blocksize
        self.samplingRate = samplingrate
        self.numChannels = numchannels
        self.numSamples = int(self.blocksize \
                              * np.ceil(self.samplingRate \
                                        / self.integ['fast'] \
                                        / self.blocksize))
        self.dummyDataIn = np.empty((self.numSamples, self.numChannels),
                                  dtype='float32')
        self.dummyDataOut = np.empty((self.numSamples, self.numChannels),
                                    dtype='float32')

        self.dummyCounter = int()
        return

    def stdout_print_dbfs(self, indata: np.ndarray,
                          outdata: np.ndarray,
                          frames: int,
                          status: sd.CallbackFlags):
        """
        Standard output print sound full scale level:
        ----------------------------------------------

            Prints the Full Scale level in decibel of the incoming sound at every 125 ms

        :param data: the audio data from opened stream
        :type data: numpy.ndarray
        :param frames: the number of samples (rows) for each column
        :type frames: int
        :param status: A condition about incoming and/or outgoing data
        :type status: sounddevice.CallbackFlags
        :return: Nothing for a "Continue" and a flag for stopping or aborting
        :rtype: sounddevice.CallbackStop or sounddevice.CallbackAbort

        Buffers enough samples to make approximately 125 ms of signal and calculate
        the root-mean-square of the buffer, then resets it
        """
        if status:
            print(status)
        elif frames != self.blocksize:
            raise ValueError("Doomsy's blocksize shoulds bee equals to streamsy's")
        try:
            self.dummyDataIn[self.dummyCounter:frames + self.dummyCounter, :] \
                = indata[:] if indata is not None else 1e-13
            self.dummyDataOut[self.dummyCounter:frames + self.dummyCounter, :] \
                = outdata[:] if outdata is not None else 1e-13
            self.dummyCounter += frames
        except ValueError:
            print(f"SPL: {20 * np.log10((np.mean(self.dummyDataIn ** 2, axis=0)) ** 0.5)} {20 * np.log10((np.mean(self.dummyDataOut ** 2, axis=0)) ** 0.5)}",
                  end='\r')
            self.dummyCounter = 0
        return

def rec(msmnt: Measurement=None, monitor: Callable=None, bs: int=32):
    if msmnt is None:
        msmnt = generate.measurement('rec')
    with Recorder(msmnt, 'float32', bs) as rec:  # Creates context with Recorder object
        if monitor is not None:
            rec.set_monitoring(monitor)  # Sets the monitor function
        rec.run()    # start to record
        signal = rec.retrieve()
    return signal

def play(msmnt: Measurement=None, monitor: Callable=None, bs: int=32):
    if msmnt is None:
        msmnt = generate.measurement('playrec')
    with Player(msmnt, 'float32', bs) as player:  # Creates context with Player object
        if monitor is not None:
            player.set_monitoring(monitor)  # Sets the monitor function
        player.run()    # start to reproduce
    return

def playrec(msmnt: Measurement=None, monitor: Callable=None, bs: int=32):
    if msmnt is None:
        msmnt = generate.measurement('playrec')
    with PlaybackRecorder(msmnt, 'float32', bs) as pbr:  # Creates context with Player object
        if monitor is not None:
            pbr.set_monitoring(monitor)  # Sets the monitor function
        pbr.run()    # start to reproduce
        signal = pbr.retrieve()
    return signal


if __name__ == "__main__":
    from pytta import generate, Recorder, SignalObj

    bs = 512

    measure = generate.measurement(device=0)  # generates a default RecMeasure object
    doomsy = DoomyDoode(measure.samplingRate,  # Generates a DoomyDoode instance
                        measure.numInChannels,
                        bs)  # blocksize

    signal = playrec(monitor=doomsy.stdout_print_dbfs, bs=bs)
    signal.plot_time()
    signal.plot_freq()
