# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd

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
    """
    def __init__(self, samplingrate, numchannels, blocksize):
        self.blocksize = blocksize
        self.samplingRate = samplingrate
        self.numChannels = numchannels
        self.numSamples = int(self.blocksize \
                              * np.ceil(self.samplingRate \
                                        / self.integ['fast'] \
                                        / self.blocksize))
        self.dummyData = np.empty((self.numSamples, self.numChannels),
                                  dtype='float32')
        self.dummyCounter = int()
        return

    def stdout_print_spl(self, data: np.ndarray, frames: int,
                             status: sd.CallbackFlags):
        """
        Standard output print sound pressure level:
        --------------------------------------------

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
        if self.dummyCounter >= self.numSamples:
            print("SPL:", 20 * np.log10((np.mean(data ** 2, axis=0)) ** 0.5))
            self.dummyCounter = 0
        else:
            self.dummyData[self.dummyCounter:frames + self.dummyCounter, :] = data[:]
            self.dummyCounter += frames
        return


if __name__ == "__main__":
    from pytta import generate, Recorder, SignalObj
    recmeasure = generate.measurement('rec')
    doomsy = DoomyDoode(recmeasure.samplingRate,
                        recmeasure.numInChannels,
                        32)  # blocksize

    with Recorder(recmeasure, 'float32', 32) as rec:
        rec.set_monitoring(doomsy.stdout_print_spl)
        rec.run()
        signal = SignalObj(rec.recData, 'time', rec.samplingRate,
                           freqMin=20, freqMax=20e3)

    signal.plot_time()
    signal.plot_freq()
