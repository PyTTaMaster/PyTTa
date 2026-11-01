"""
Tests for extrapolate of pytta.functions, which follows ita_extrapolate of
the ITA-Toolbox with IIR fits by invfreqz in place of MATLAB's rationalfit.
"""
import unittest

import numpy as np
import scipy.signal as ss

import pytta

FS = 48000


class TestExtrapolate(unittest.TestCase):

    def test_rational_response_is_extrapolated_exactly(self):
        # A band-pass of order 4 known only between 300 Hz and 3 kHz (the
        # rest of the spectrum replaced by noise) comes back outside that
        # range from models of the same order, within what the float32
        # samples of SignalObj allow.
        label = 'x'
        n = 2 ** 14
        b, a = ss.butter(2, [100, 5000], 'bandpass', fs=FS)
        h = ss.lfilter(b, a, np.eye(1, n)[0])
        H = np.fft.rfft(h)
        f = np.fft.rfftfreq(n, 1 / FS)
        outside = (f < 300) | (f > 3000)
        rng = np.random.default_rng(7)
        corrupted = H.copy()
        corrupted[outside] = rng.standard_normal(outside.sum()) + 1j * rng.standard_normal(outside.sum())
        y = pytta.extrapolate(pytta.SignalObj(np.fft.irfft(corrupted, n), 'time', FS), [300, 3000], 4)
        Y = np.fft.rfft(np.asarray(y.timeSignal[:, 0], dtype=float))
        e = np.abs(Y[outside] - H[outside]).max() / np.abs(H).max()
        print(f'\n  extrapolation outside 300 Hz to 3 kHz: {e:.1e}', end='')
        self.assertLess(e, 1e-3)


if __name__ == '__main__':
    unittest.main()
