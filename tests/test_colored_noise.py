"""
Tests for pytta.generate.colored_noise.

A noise of power law B has a power spectral density proportional to
1 / f**B: -3.01 B dB per octave (white 0, pink -3.01, red -6.02, blue
+3.01, purple +6.02). Each test pins one rule on noise generated with a
fixed seed; the slope is fitted to the mean PSD of octave bands from 63 Hz
to 8 kHz, averaged over eight channels.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000
COLORS = {'purple': -2, 'blue': -1, 'white': 0, 'pink': 1, 'red': 2}


def octave_slope(signalObj):
    psd = np.mean(np.abs(signalObj.freqSignal) ** 2, axis=1)
    f = signalObj.freqVector
    centres = 63 * 2.0 ** np.arange(8)
    levels = [10 * np.log10(psd[(f >= c / 2 ** .5) & (f < c * 2 ** .5)].mean())
              for c in centres]
    return np.polyfit(np.log2(centres), levels, 1)[0]


class TestColoredNoise(unittest.TestCase):

    def test_spectrum_follows_the_power_law_of_the_color(self):
        # With the default kind of use: silent margins around the noise.
        for color, B in COLORS.items():
            np.random.seed(0)
            noise = pytta.generate.colored_noise(
                color, samplingRate=FS, fftDegree=16, numChannels=8,
                startMargin=0.1, stopMargin=0.1)
            slope = octave_slope(noise)
            print(f'\n  {color:6s} {slope:+.2f} dB/oct (expected '
                  f'{-3.0103 * B:+.2f})')
            self.assertAlmostEqual(slope, -3.0103 * B, delta=0.2)

    def test_noise_has_no_dc_offset(self):
        np.random.seed(0)
        noise = pytta.generate.colored_noise(
            'pink', samplingRate=FS, fftDegree=16, numChannels=4,
            startMargin=0, stopMargin=0).timeSignal
        offset = np.max(np.abs(noise.mean(axis=0)) / noise.std(axis=0))
        print(f'\n  |mean| / rms = {offset:.3f}')
        self.assertLess(offset, 0.05)

    def test_every_channel_has_random_phase(self):
        # A spectrum with no imaginary part gives a noise that is its own
        # mirror image in time, x[n] = x[N - n].
        np.random.seed(0)
        noise = pytta.generate.colored_noise(
            'pink', samplingRate=FS, fftDegree=14, numChannels=2,
            startMargin=0, stopMargin=0).timeSignal
        N = noise.shape[0]
        k = np.arange(N // 10, N - N // 10)
        for ch in range(2):
            r = np.corrcoef(noise[k, ch], noise[N - k, ch])[0, 1]
            print(f'\n  channel {ch}: corr(x[n], x[N-n]) = {r:+.3f}')
            self.assertLess(abs(r), 0.2)


if __name__ == '__main__':
    unittest.main()
