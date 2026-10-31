"""
Tests for the energy decay curve methods and the reverberation time
uncertainty of pytta.RoomAnalysis.

The impulse response is a synthetic exponential decay of white noise with
T = 1 s, over a steady background noise 45 dB below its start, so the
methods of noise compensation of ISO 3382-1 (Guski and Vorlaender, 2013)
can be told apart. Each test pins one rule.
"""
import os
import unittest
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000
T = 1.0


def decay_with_noise():
    rng = np.random.default_rng(0)
    t = np.arange(int(2.5 * FS)) / FS
    h = rng.standard_normal(t.size) * 10 ** (-3 * t / T) \
        + 10 ** (-45 / 20) * rng.standard_normal(t.size)
    return pytta.SignalObj(h, 'time', FS, signalType='energy')


def analyse(ir, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pytta.RoomAnalysis(ir, nthOct=1, minFreq=500., maxFreq=2000.,
                                  **kwargs)


class TestRoomEDC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ir = decay_with_noise()
        cls.T30 = {m: analyse(ir, edcMethod=m).T30 for m in
                   ('cutWithCorrection', 'justCut', 'noCut', 'subtractNoise')}
        cls.bypass = analyse(ir, bypassLundeby=True)
        for m, v in cls.T30.items():
            print(f'\n  {m:18s} T30 = {np.round(v, 3)}')

    def test_compensated_methods_find_the_true_decay(self):
        for m in ('cutWithCorrection', 'subtractNoise'):
            self.assertTrue(np.allclose(self.T30[m], T, rtol=0.05), m)

    def test_uncompensated_methods_are_biased_as_expected(self):
        # The background noise flattens the tail of the full curve, and a
        # truncation without correction makes the curve fall too fast.
        self.assertTrue(np.all(self.T30['noCut'] > 1.10 * T))
        self.assertTrue(np.all(self.T30['justCut']
                               < self.T30['cutWithCorrection']))

    def test_bypass_lundeby_is_the_no_cut_method(self):
        self.assertTrue(np.array_equal(self.bypass.T30, self.T30['noCut']))

    def test_uncertainty_follows_iso_3382_1(self):
        # ISO 3382-1:2009, Equations (4) and (5), with the bandwidth of an
        # octave filter B = 0.71 fc and n = 10 for an impulse response
        # (clause 7.2).
        ra = analyse(decay_with_noise())
        fc = np.array([500., 1000., 2000.])
        for param, k, c in (('T20', 0.88, 1.90), ('T30', 0.55, 1.52)):
            rt = getattr(ra, param)
            for N in (1, 4):
                iso = k * rt * np.sqrt((1 + c / 10) / (N * 0.71 * fc * rt))
                got = ra.uncertainty(param, numPositions=N)
                self.assertTrue(np.allclose(got, iso, rtol=0.005), param)
        with self.assertRaises(ValueError):
            ra.uncertainty('EDT')


if __name__ == '__main__':
    unittest.main()
