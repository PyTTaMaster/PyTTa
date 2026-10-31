#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of the functions of pytta.functions, which are also available as
pytta.<function>. Run it as a script; the figures open at the end.
"""
import os
import tempfile

import matplotlib.pyplot as plt

import pytta

if __name__ == "__main__":

    # Signals to work with: two sines and a sweep, at 48 kHz
    sine1k = pytta.generate.sin(Arms=0.5, freq=1000, timeLength=1,
                                samplingRate=48000)
    sine2k = pytta.generate.sin(Arms=0.25, freq=2000, timeLength=1,
                                samplingRate=48000)
    sweep = pytta.generate.sweep(freqMin=50, freqMax=16000, fftDegree=16,
                                 samplingRate=48000)

    # Audio devices available to measure with
    print(pytta.list_devices())

    # Several signals as channels of one SignalObj, and back
    twoChannels = pytta.merge(sine1k, sine2k)
    first, second = pytta.split(twoChannels)
    print('channels:', twoChannels.numChannels, '->', first.numChannels,
          'and', second.numChannels)

    # Convolution, here of the sweep with a 10 ms delay of half amplitude
    delayed = pytta.SignalObj([0.0] * 480 + [0.5], 'time', 48000)
    echo = pytta.fft_convolve(sweep, delayed)

    # Delay between two signals, in samples, and their correlation
    print('delay of the echo:', pytta.find_delay(sweep, echo), 'samples')
    print('correlation of a sine with itself:',
          pytta.corr_coef(sine1k, sine1k))

    # Change of sampling rate
    sine44 = pytta.resample(sine1k, 44100)
    print('samples at 44.1 kHz:', sine44.numSamples)

    # Time of the peak of each signal
    print('peak of the echo at', pytta.peak_time(echo), 's')

    # Sound pressure level in third octave bands, as an Analysis
    level = pytta.SPL(sine1k, nthOct=3, minFreq=500, maxFreq=2000)
    print(level)

    # Wave files and HDF5 files (PyTTa objects keep their metadata)
    with tempfile.TemporaryDirectory() as folder:
        wavName = os.path.join(folder, 'sine.wav')
        pytta.write_wav(wavName, sine1k)
        fromWav = pytta.read_wav(wavName)
        h5Name = os.path.join(folder, 'signals.hdf5')
        pytta.save(h5Name, sine1k, sweep)
        loaded = pytta.load(h5Name)
        print('read from the wave file:', fromWav.numSamples, 'samples;',
              'loaded from the HDF5 file:', list(loaded.keys()))

    # Figures
    pytta.plot_time(sine1k, sine2k)
    pytta.plot_time_dB(echo)
    pytta.plot_freq(sweep, smooth=True)
    pytta.plot_bars(level)
    pytta.plot_spectrogram(sweep)
    plt.show()
