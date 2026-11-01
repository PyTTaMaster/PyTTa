"""
Tests for peak_time with a negative peak and for the regularized
deconvolution of ImpulsiveResponse with short signals, whose crossfade
below freqMin is narrower than one frequency bin.
"""
import os
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

FS = 48000


def signal(*samples, **kwargs):
    # A tuple as first local variable: SignalObj compares it with itself
    # while walking the stack, which an array cannot do before #59.
    return pytta.SignalObj(samples[0], 'time', kwargs.pop('fs', FS), **kwargs)


class TestPeakAndShortRegularization(unittest.TestCase):

    def test_peak_time_of_a_negative_peak(self):
        x = np.zeros((1000, 2))
        x[300, 0], x[100, 0] = -1, 0.5          # negative peak at 0.3 s
        x[700, 1] = 1                           # positive peak at 0.7 s
        np.testing.assert_allclose(pytta.peak_time(signal(x, fs=1000)), [0.3, 0.7])

    def test_short_signals_deconvolve(self):
        # 4096 samples at 48 kHz: bins of 11.7 Hz, wider than the crossfade
        # from freqMin / sqrt(2) to freqMin. The regularized response equals
        # the known one between 100 Hz and 10 kHz.
        rng = np.random.default_rng(3)
        x = rng.standard_normal(4096) * 0.1
        h = np.exp(-np.arange(200) / 30) * rng.standard_normal(200)
        y = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h, 4096), 4096)
        ir = pytta.ImpulsiveResponse(signal(x), signal(y), 'linear')
        f = np.fft.rfftfreq(4096, 1 / FS)
        band = (f > 100) & (f < 10000)
        H = np.fft.rfft(h, 4096)
        M = np.fft.rfft(np.asarray(ir.IR.timeSignal[:, 0], dtype=float))
        e = np.abs(M[band] - H[band]).max() / np.abs(H[band]).max()
        print(f'\n  short regularized deconvolution: {e:.1e}', end='')
        self.assertLess(e, 1e-4)        # float32 samples


if __name__ == '__main__':
    unittest.main()
