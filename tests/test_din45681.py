"""
Tests for pytta.din45681, the tonal audibility of DIN 45681:2002 as
computed by ita_tonality of the ITA-Toolbox. Each test pins one rule.

The ITA golden data come from golden/make_din45681_ita.m. The noise there
is uniform noise from MATLAB's Mersenne twister, which
numpy.random.RandomState rebuilds; the stored sums check the samples.

SignalObj inspects the caller's frame (classes/_instanceinfo.py) and fails
when the caller's first local variable is an array, so the objects are
built after a scalar local.
"""
import os
import unittest

import numpy as np
import scipy.io as sio

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta
from pytta import din45681

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
P0 = 2e-5


def golden_signal(case, fs):
    """Rebuild the signal of one case of make_din45681_ita.m."""
    t = np.arange(3 * fs) / fs
    noise = np.random.RandomState(int(case.seed)).random_sample(t.size) - 0.5
    x = case.noise_rms * noise / np.sqrt(1 / 12)
    for f, L in zip(np.atleast_1d(case.tones), np.atleast_1d(case.levels)):
        x = x + np.sqrt(2) * P0 * 10**(L / 20) * np.sin(2 * np.pi * f * t)
    return x.astype(np.float32).astype(float)


def a_weighting(f):
    return 20 * np.log10(12194**2 * f**4 / ((f**2 + 20.6**2) * np.sqrt(
        (f**2 + 107.7**2) * (f**2 + 737.9**2)) * (f**2 + 12194**2))) + 2.0


class TestTonalAudibility(unittest.TestCase):

    def test_tone_in_white_noise_against_the_formula(self):
        # DeltaL = LT - LG - av, LG = LS + 10 lg(dfc / df), with LS the
        # level of white noise in one 2 Hz line and av = -2 - lg(1 +
        # (f/502)^2.5). At 1 kHz the A weighting is 0 dB, so the tone keeps
        # its level; the noise lines around it keep theirs within 0.1 dB.
        # LS is estimated from about 80 lines of 19 blocks: over 8 noise
        # seeds it scatters from -0.22 to +0.34 dB, hence 0.5 dB.
        label = 'tone in noise'
        fs, f, L, sigma = 48000, 1000.0, 60.0, 0.02
        t = np.arange(4 * fs) / fs
        x = (sigma * np.random.RandomState(7).standard_normal(t.size)
             + np.sqrt(2) * P0 * 10**(L / 20) * np.sin(2 * np.pi * f * t))
        out = din45681.tonal_audibility(pytta.SignalObj(x, 'time', fs))
        N = fs // 2  # lines of 2 Hz; one line holds 2 sigma^2 / N of the noise
        LS = 10 * np.log10(2 * sigma**2 / N / P0**2) + a_weighting(f)
        dfc = 25 + 75 * (1 + 1.4 * (f / 1000)**2)**0.69
        expected = L - (LS + 10 * np.log10(dfc / 2)) + 2 + np.log10(1 + (f / 502)**2.5)
        print(f'\n  DeltaL = {out["deltaL"]:.3f} dB at {out["frequency"]:.0f} Hz, '
              f'expected {expected:.3f} dB, diff {out["deltaL"] - expected:+.3f}')
        self.assertAlmostEqual(out['frequency'], f, delta=2)
        self.assertAlmostEqual(out['deltaL'], expected, delta=0.5)

    def test_matches_ita_tonality(self):
        # Same samples in both. ita_tonality reads every frequency 2 Hz high
        # (its frequency vector starts at 2 Hz while the first FFT line is
        # 0 Hz), so its frequencies are compared less 2 Hz. The rest of the
        # method is the same; the A weighting read 2 Hz off shifts DeltaL by
        # far less than 0.05 dB.
        d = sio.loadmat(os.path.join(GOLDEN, 'din45681_ita.mat'),
                        squeeze_me=True, struct_as_record=False)
        fs = int(d['fs'])
        for case in np.atleast_1d(d['cases']):
            label = case.name
            x = golden_signal(case, fs)
            self.assertAlmostEqual(np.sum(x), case.sum, delta=1e-9)
            out = din45681.tonal_audibility(pytta.SignalObj(x, 'time', fs))
            info = np.atleast_2d(case.info)
            print(f'\n  {case.name}: DeltaL {out["deltaL"]:.3f} vs {case.deltaL:.3f} dB, '
                  f'{out["frequency"]:.0f} vs {case.frequency - 2:.0f} Hz, '
                  f'{len(out["tones"])} vs {info.shape[0]} tones', end='')
            self.assertAlmostEqual(out['deltaL'], case.deltaL, delta=0.05)
            self.assertAlmostEqual(out['frequency'], case.frequency - 2, delta=1e-9)
            self.assertEqual(len(out['tones']), info.shape[0])

    def test_noise_has_no_tone(self):
        label = 'noise'
        fs = 48000
        x = 0.02 * np.random.RandomState(3).standard_normal(4 * fs)
        out = din45681.tonal_audibility(pytta.SignalObj(x, 'time', fs))
        print(f'\n  {len(out["tones"])} tones, DeltaL {out["deltaL"]}')
        self.assertEqual(out['tones'], [])
        self.assertEqual(out['deltaL'], 0)

    def test_rejects_invalid_input(self):
        label = 'invalid'
        fs = 48000
        with self.assertRaises(ValueError):
            din45681.tonal_audibility(pytta.SignalObj(np.zeros((4 * fs, 2)), 'time', fs))
        with self.assertRaises(ValueError):
            din45681.tonal_audibility(pytta.SignalObj(np.zeros(fs // 4), 'time', fs))
        with self.assertRaises(TypeError):
            din45681.tonal_audibility(np.zeros(4 * fs))


if __name__ == '__main__':
    unittest.main()
