"""
Tests for pytta.minimum_phase.

The minimum phase version of a signal has the same magnitude spectrum and
the phase that puts its energy as early as possible. Each test pins one
rule. The first uses a short filter whose minimum phase version is known
exactly: zeros inside the unit circle give a minimum phase filter, and its
time reversal has the same magnitude and maximum phase.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000
N = 1024
TOL32 = 1e-6  # rounding of the float32 SignalObj storage


class TestMinimumPhase(unittest.TestCase):

    def test_known_filter_comes_back_in_minimum_phase(self):
        b = np.real(np.poly([0.9, -0.7, 0.5 * np.exp(0.4j),
                             0.5 * np.exp(-0.4j)]))
        data = np.zeros((N, 2))
        data[:b.size, 0] = b          # minimum phase already
        data[:b.size, 1] = b[::-1]    # same magnitude, maximum phase
        sig = pytta.SignalObj(data, 'time', FS)
        out = pytta.minimum_phase(sig).timeSignal
        for ch in range(2):
            err = np.max(np.abs(out[:, ch] - data[:, 0]))
            print(f'\n  channel {ch}: max |error| = {err:.1e}')
            self.assertLess(err, TOL32)

    def test_magnitude_is_kept(self):
        # A delayed, decaying noise burst: far from minimum phase.
        rng = np.random.default_rng(0)
        t = np.arange(N)
        x = rng.standard_normal(N) * np.exp(-(t - 100) / 80) * (t >= 100)
        sig = pytta.SignalObj(x, 'time', FS)
        out = pytta.minimum_phase(sig)
        X = np.abs(np.fft.rfft(sig.timeSignal[:, 0]))
        Y = np.abs(np.fft.rfft(out.timeSignal[:, 0]))
        loud = X > 1e-3 * X.max()
        err = np.max(np.abs(Y[loud] / X[loud] - 1))
        print(f'\n  max relative magnitude change = {err:.1e}')
        self.assertLess(err, 1e-5)

    def test_energy_comes_as_early_as_possible(self):
        # The cumulative energy of the minimum phase version is at least
        # that of the input at every sample. The cepstrum is computed on the
        # signal length, so its time aliasing allows a small deficit:
        # 2e-6 of the total energy on this signal, 3e-4 on the burst of
        # test_magnitude_is_kept.
        rng = np.random.default_rng(1)
        t = np.arange(N)
        x = rng.standard_normal(N) * np.exp(-(t - 300) / 60) * (t >= 300)
        sig = pytta.SignalObj(x, 'time', FS)
        y = pytta.minimum_phase(sig).timeSignal[:, 0].astype(float)
        x = sig.timeSignal[:, 0].astype(float)
        deficit = np.max(np.cumsum(x ** 2) - np.cumsum(y ** 2)) / np.sum(x ** 2)
        print(f'\n  largest energy deficit = {deficit:.1e} of the total')
        self.assertLess(deficit, 1e-3)
        self.assertLess(np.argmax(np.abs(y)), 50)

    def test_returns_new_object_and_keeps_metadata(self):
        rng = np.random.default_rng(2)
        sig = pytta.SignalObj(rng.standard_normal((N, 2)), 'time', FS,
                              signalType='energy')
        before = sig.timeSignal.copy()
        out = pytta.minimum_phase(sig)
        self.assertTrue(np.array_equal(sig.timeSignal, before))
        self.assertIsNot(out, sig)
        self.assertEqual((out.samplingRate, out.numSamples, out.numChannels,
                          out.signalType), (FS, N, 2, 'energy'))
        with self.assertRaises(TypeError):
            pytta.minimum_phase(before)


if __name__ == '__main__':
    unittest.main()
