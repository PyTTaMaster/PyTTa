"""
Tests for pytta.generate.multitone and the functions crossfade_spectra,
cut_impulse_train and velocity_from_pressure_gradient of pytta.functions,
which follow the ITA-Toolbox (ita_multitone, ita_xfade_spk,
ita_cutImpulseTrain, ita_velocity_from_pressure_gradient). Each test pins
one rule against an analytic result or golden data of the ITA-Toolbox from
tests/golden/make_cut_impulse_train_ita.m (MATLAB R2026b).
"""
import os
import unittest

import numpy as np
from scipy.io import loadmat

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


def tone(f, fs, n, phase=0):
    return np.cos(2 * np.pi * f * np.arange(n) / fs + phase)


class TestDSPMisc(unittest.TestCase):

    def test_multitone_tones_fall_on_bins(self):
        # Every tone has a whole number of periods, so the DFT is A N / 2 at
        # the rounded bins and zero elsewhere.
        label = 'x'
        fs, deg = 48000, 14
        n = 2 ** deg
        sig = pytta.generate.multitone([100, 316.2, 1000, 3162, 10000], 0.3, samplingRate=fs, fftDegree=deg)
        X = np.abs(np.fft.rfft(sig.timeSignal[:, 0].astype(float)))
        bins = np.round(np.array([100, 316.2, 1000, 3162, 10000]) * n / fs).astype(int)
        e_on = np.abs(X[bins] - 0.3 * n / 2).max() / (0.3 * n / 2)
        e_off = np.delete(X, bins).max() / (0.3 * n / 2)
        print(f'\n  multitone: on bins {e_on:.1e}, off bins {e_off:.1e}', end='')
        self.assertLess(e_on, 1e-6)
        self.assertLess(e_off, 1e-6)

    def test_crossfade_keeps_each_band_and_sums_to_one(self):
        # Below f0 the low signal, above f1 the high one, and the raised
        # cosine weights add up to 1, so a signal crossfaded with itself
        # comes back.
        label = 'x'
        fs, n = 8000, 4096
        rng = np.random.default_rng(3)
        a = pytta.SignalObj(rng.standard_normal(n), 'time', fs)
        b = pytta.SignalObj(rng.standard_normal(n), 'time', fs)
        A, B = (np.fft.rfft(s.timeSignal[:, 0].astype(float)) for s in (a, b))
        X = np.fft.rfft(pytta.crossfade_spectra(a, b, [500, 1000]).timeSignal[:, 0].astype(float))
        b0, b1 = 500 * n // fs, 1000 * n // fs
        e_low = np.abs(X[:b0] - A[:b0]).max() / np.abs(A).max()
        e_high = np.abs(X[b1 + 1:] - B[b1 + 1:]).max() / np.abs(B).max()
        same = pytta.crossfade_spectra(a, a, [500, 1000]).timeSignal - a.timeSignal
        print(f'\n  crossfade: low band {e_low:.1e}, high band {e_high:.1e}, '
              f'with itself {np.abs(same).max():.1e}', end='')
        self.assertLess(max(e_low, e_high), 1e-6)
        self.assertLess(np.abs(same).max(), 1e-6)

    def test_cut_impulse_train_matches_ita(self):
        g = loadmat(os.path.join(GOLDEN, 'cut_impulse_train_ita.mat'))
        sig = pytta.SignalObj(g['x'].astype(float), 'time', g['fs'].item())
        cuts, starts = pytta.cut_impulse_train(sig, length=0.25, start_before=0.002)
        print(f'\n  cut_impulse_train: starts {starts}, ITA {g["start"].ravel() - 1}', end='')
        np.testing.assert_array_equal(starts, g['start'].ravel() - 1)
        np.testing.assert_array_equal(cuts[0].timeSignal, g['c1'])
        np.testing.assert_array_equal(cuts[1].timeSignal, g['c2'])

    def test_pressure_gradient_of_a_plane_wave(self):
        # For a plane wave the finite difference gives the velocity at the
        # midpoint, p / (rho c), times sin(kd/2) / (kd/2).
        label = 'x'
        fs, n, f, d, rho, c = 48000, 4800, 1000, 0.05, 1.2, 343
        k = 2 * np.pi * f / c
        p = pytta.SignalObj(np.column_stack([tone(f, fs, n), tone(f, fs, n, -k * d)]), 'time', fs)
        v = pytta.velocity_from_pressure_gradient(p, d, rho, c).timeSignal[:, 0]
        expected = np.sinc(k * d / 2 / np.pi) * tone(f, fs, n, -k * d / 2) / (rho * c)
        e = np.abs(v - expected).max() * rho * c
        print(f'\n  pressure gradient: {e:.1e} of p / (rho c)', end='')
        self.assertLess(e, 1e-6)


if __name__ == '__main__':
    unittest.main()
