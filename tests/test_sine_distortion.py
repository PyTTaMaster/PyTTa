"""
Tests for pytta.sine_distortion and pytta.snr, for measurements with a
sine, such as those of line inputs and outputs. Each test pins one rule on
signals whose distortion and noise are known.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000
F0 = 997.0                 # off the bin grid, as test signals usually are
tt = np.arange(FS) / FS


class TestSineDistortion(unittest.TestCase):

    def test_harmonics_give_the_thd(self):
        # Second and third harmonics at 1 % and 0.5 % of the fundamental:
        # THD = sqrt(0.01**2 + 0.005**2) re the fundamental, and THD+N the
        # same rms re the total rms.
        x = np.sin(2 * np.pi * F0 * tt) + 0.01 * np.sin(4 * np.pi * F0 * tt) \
            + 0.005 * np.sin(6 * np.pi * F0 * tt)
        thd, thdn = pytta.sine_distortion(pytta.SignalObj(x, 'time', FS), F0)
        expected = np.hypot(0.01, 0.005)
        print(f'\n  THD {thd[0]:.6f}, THD+N {thdn[0]:.6f} '
              f'(expected {expected:.6f})')
        self.assertAlmostEqual(thd[0] / expected, 1, delta=1e-3)
        self.assertAlmostEqual(thdn[0] / (expected / np.sqrt(1 + expected ** 2)),
                               1, delta=1e-3)

    def test_noise_enters_thd_plus_n_and_not_thd(self):
        # White noise of rms 0.01 on a sine of rms 1/sqrt(2): THD+N is the
        # noise rms re the total rms, while THD only sees the noise in the
        # narrow bands of the harmonics.
        rng = np.random.default_rng(0)
        noise = 0.01 * rng.standard_normal(FS)
        x = np.sin(2 * np.pi * F0 * tt) + noise
        thd, thdn = pytta.sine_distortion(pytta.SignalObj(x, 'time', FS), F0)
        expected = np.std(noise) / np.sqrt(0.5 + np.var(noise))
        print(f'\n  THD+N {thdn[0]:.5f} (expected {expected:.5f}), '
              f'THD {thd[0]:.5f}')
        self.assertAlmostEqual(thdn[0] / expected, 1, delta=0.01)
        self.assertLess(thd[0], expected / 10)

    def test_snr_of_signal_and_noise_recordings(self):
        rng = np.random.default_rng(1)
        sig = pytta.SignalObj(np.sin(2 * np.pi * F0 * tt), 'time', FS)
        noise = pytta.SignalObj(1e-3 * rng.standard_normal(FS), 'time', FS)
        expected = 20 * np.log10(np.sqrt(0.5) / np.std(noise.timeSignal))
        self.assertAlmostEqual(pytta.snr(sig, noise)[0], expected, delta=0.01)

    def test_rejects_a_fundamental_beyond_nyquist(self):
        sig = pytta.SignalObj(np.zeros(FS), 'time', FS)
        with self.assertRaises(ValueError):
            pytta.sine_distortion(sig, FS / 2)


if __name__ == '__main__':
    unittest.main()
