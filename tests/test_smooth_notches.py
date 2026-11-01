"""
Tests for smooth_notches of pytta.functions, which follows
ita_smooth_notches of the ITA-Toolbox ("compare and squeeze"). Each test
pins one rule on the spectrum of a peaking filter.
"""
import os
import unittest

import numpy as np
import scipy.signal as ss

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

FS = 48000


def peaking(*gain):
    # Impulse response of a peaking biquad at 1 kHz, Q 8 (Bristow-Johnson).
    label = 'x'
    A = 10 ** (gain[0] / 40)
    w0 = 2 * np.pi * 1000 / FS
    alpha = np.sin(w0) / 16
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return pytta.SignalObj(ss.lfilter(b, a, np.eye(1, 2 ** 15)[0]), 'time', FS)


def level(signal):
    return 20 * np.log10(np.abs(np.asarray(signal.freqSignal)[:, 0]))


class TestSmoothNotches(unittest.TestCase):

    def test_deep_notch_keeps_squeeze_times_its_depth(self):
        # In the notch the level becomes Ls + 0.3 (L - Ls), Ls the smoothed
        # level; elsewhere the spectrum is untouched.
        x = peaking(-20)
        out = pytta.smooth_notches(x, 3, 3, 0.3)
        L, Ls = level(x), level(pytta.smooth_frequency(x, 3))
        k = np.argmin(L)
        expected = Ls[k] + 0.3 * (L[k] - Ls[k])
        print(f'\n  notch: {L[k]:.2f} dB -> {level(out)[k]:.2f} dB, expected {expected:.2f} dB', end='')
        self.assertAlmostEqual(level(out)[k], expected, places=4)   # float32 spectrum
        f = x.freqVector
        far = (f > 3000) | ((f < 300) & (f > 0))
        np.testing.assert_allclose(level(out)[far], L[far], atol=1e-4)   # float32

    def test_shallow_dip_is_left_alone(self):
        # A dip of 2 dB stays below the threshold of 3 dB.
        x = peaking(-2)
        out = pytta.smooth_notches(x, 3, 3, 0.3)
        np.testing.assert_allclose(level(out), level(x), atol=1e-4)   # float32


if __name__ == '__main__':
    unittest.main()
