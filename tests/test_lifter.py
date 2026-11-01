"""
Tests for lifter of pytta.functions, which follows ita_lifter of the
ITA-Toolbox. The cepstrum of an echo lies only at multiples of its delay,
which gives exact results.
"""
import os
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

FS = 48000


def echo(*delay):
    label = 'x'
    h = np.zeros(4096)
    h[0], h[delay[0]] = 1, 0.5
    return pytta.SignalObj(h, 'time', FS)


class TestLifter(unittest.TestCase):

    def test_short_lifter_removes_the_echo(self):
        # The echo at 100 samples ripples the log spectrum with cepstral
        # peaks at 100, 200, ...; a window of 50 quefrencies leaves the
        # impulse alone, with either taper.
        for alpha in (0, 1):
            y = np.asarray(pytta.lifter(echo(100), 50, alpha).timeSignal[:, 0], dtype=float)
            e = np.abs(y - np.eye(1, 4096)[0]).max()
            print(f'\n  lifter, alpha {alpha}: {e:.1e} off the impulse', end='')
            self.assertLess(e, 1e-6)

    def test_long_lifter_keeps_the_echo(self):
        # A rectangular window reaching past the cepstrum leaves the
        # response as it was.
        x = np.asarray(echo(100).timeSignal[:, 0], dtype=float)
        y = np.asarray(pytta.lifter(echo(100), 2000).timeSignal[:, 0], dtype=float)
        print(f'\n  long lifter: {np.abs(y - x).max():.1e}', end='')
        self.assertLess(np.abs(y - x).max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
