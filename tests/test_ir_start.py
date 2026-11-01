"""
Tests for the start of an impulse response that RoomAnalysis and crop_IR
take from the broadband response (ISO 3382-1:2009, A.3.4): the last sample
before the maximum that lies 20 dB below it, with 0-based indices.
"""
import os
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

from pytta.classes.analysis import _start_sample_ISO3382

FS = 48000


def response(onset):
    # Silence, then a peak and an exponential decay to a noise floor 60 dB
    # down, so that the noise check of the function passes.
    rng = np.random.default_rng(0)
    n = FS
    x = 1e-3 * rng.standard_normal(n)
    t = np.arange(n - onset) / FS
    x[onset:] += np.exp(-6.9 * t / 0.3) * np.sign(rng.standard_normal(t.size))
    x[onset] = 1.
    return x


class TestIRStart(unittest.TestCase):

    def test_peak_on_the_first_sample(self):
        # Nothing comes before the peak, so the response starts at sample 0;
        # starting at 1 dropped the peak itself.
        self.assertEqual(_start_sample_ISO3382(response(0), 20), 0)

    def test_last_sample_below_the_threshold(self):
        # A ramp from -40 dB to the peak at sample 100: the samples from 90
        # on lie within 20 dB of it, so the start is sample 89.
        x = response(100)
        x[:100] = 10 ** (np.linspace(-40, -0.5, 100) / 20)
        expected = np.flatnonzero(20 * np.log10(np.abs(x[:100])) < -20)[-1]
        self.assertEqual(_start_sample_ISO3382(x, 20), expected)

    def test_only_the_first_sample_below_the_threshold(self):
        # The first sample is the last one 20 dB below the peak: the start
        # is sample 0, where 1 was returned before.
        x = response(1)
        x[0] = 1e-3
        self.assertEqual(_start_sample_ISO3382(x, 20), 0)


if __name__ == '__main__':
    unittest.main()
