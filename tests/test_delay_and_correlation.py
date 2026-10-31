"""
Tests for pytta.find_delay and pytta.corr_coef. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000


class TestDelayAndCorrelation(unittest.TestCase):

    def test_delay_in_samples_of_the_second_signal(self):
        # Positive when the second signal comes later; the signals may have
        # different lengths.
        x = np.random.default_rng(0).standard_normal(FS)
        early = pytta.SignalObj(x, 'time', FS)
        late = pytta.SignalObj(np.concatenate((np.zeros(100), x)), 'time', FS)
        self.assertEqual(pytta.find_delay(early, late), 100)
        self.assertEqual(pytta.find_delay(late, early), -100)

    def test_correlation_coefficient_of_two_signals(self):
        # One coefficient for the first channels, not one per sample.
        x = np.random.default_rng(1).standard_normal(FS)
        a = pytta.SignalObj(x, 'time', FS)
        b = pytta.SignalObj(-2 * x, 'time', FS)
        self.assertAlmostEqual(pytta.corr_coef(a, a), 1, places=6)
        self.assertAlmostEqual(pytta.corr_coef(a, b), -1, places=6)


if __name__ == '__main__':
    unittest.main()
