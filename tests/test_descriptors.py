"""
Tests for pytta.descriptors, signal descriptors with exact values for
known signals. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta
from pytta import descriptors

FS = 48000
T = np.arange(FS) / FS


class TestDescriptors(unittest.TestCase):

    def test_crest_factor(self):
        # Peak over rms: sqrt(2) (3.01 dB) for a sine, 1 (0 dB) for a square
        # wave, per channel.
        x = np.column_stack((np.sin(2 * np.pi * 1000 * T),
                             np.sign(np.sin(2 * np.pi * 1000 * T + 0.1))))
        cf = descriptors.crest_factor(pytta.SignalObj(x, 'time', FS))
        self.assertTrue(np.allclose(cf, [np.sqrt(2), 1], rtol=1e-4))
        cf_dB = descriptors.crest_factor(pytta.SignalObj(x, 'time', FS),
                                         dB=True)
        self.assertTrue(np.allclose(cf_dB, [3.0103, 0], atol=1e-3))

    def test_kurtosis(self):
        # Pearson kurtosis: 1.5 for a sine, 3 for Gaussian noise.
        noise = np.random.default_rng(0).standard_normal(10 * FS)
        k = descriptors.kurtosis(pytta.SignalObj(noise, 'time', FS))
        self.assertAlmostEqual(k[0], 3, delta=0.02)
        k = descriptors.kurtosis(pytta.SignalObj(
            np.sin(2 * np.pi * 1000 * T), 'time', FS))
        self.assertAlmostEqual(k[0], 1.5, delta=1e-4)

    def test_spectral_shape_of_a_sine(self):
        # All the power at f: centroid f, spread 0, flatness about 0.
        sig = pytta.SignalObj(np.sin(2 * np.pi * 1000 * T), 'time', FS)
        self.assertAlmostEqual(descriptors.spectral_centroid(sig)[0], 1000,
                               delta=0.5)
        self.assertLess(descriptors.spectral_spread(sig)[0], 1)
        self.assertLess(descriptors.spectral_flatness(sig)[0], 1e-6)

    def test_spectral_shape_of_white_noise(self):
        # The power of each bin of white Gaussian noise is exponentially
        # distributed: geometric over arithmetic mean is exp(-gamma) =
        # 0.5615. The power is flat, so 85 % of it lies below 0.85 fs/2.
        noise = np.random.default_rng(1).standard_normal(10 * FS)
        sig = pytta.SignalObj(noise, 'time', FS)
        self.assertAlmostEqual(descriptors.spectral_flatness(sig)[0],
                               np.exp(-np.euler_gamma), delta=0.01)
        self.assertAlmostEqual(descriptors.spectral_rolloff(sig)[0],
                               0.85 * FS / 2, delta=0.01 * FS / 2)
        self.assertAlmostEqual(descriptors.spectral_centroid(sig)[0], FS / 4,
                               delta=0.01 * FS / 2)


if __name__ == '__main__':
    unittest.main()
