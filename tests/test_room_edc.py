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
import scipy.signal as ss

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


class TestRoomExtras(unittest.TestCase):

    def test_psnr_follows_the_noise_floor(self):
        # The same decay over background noise 45 and 55 dB below its start:
        # the peak-to-noise ratio of each band grows by 10 dB. Without the
        # Lundeby estimate (noCut) there is no noise level, so no PSNR.
        rng = np.random.default_rng(0)
        t = np.arange(int(2.5 * FS)) / FS
        decay = rng.standard_normal(t.size) * 10 ** (-3 * t / T)
        noise = rng.standard_normal(t.size)
        psnr = []
        for floor in (45, 55):
            ir = pytta.SignalObj(decay + 10 ** (-floor / 20) * noise, 'time',
                                 FS, signalType='energy')
            psnr.append(analyse(ir).PSNR)
        print(f'\n  PSNR difference by band: {np.round(psnr[1] - psnr[0], 2)}')
        self.assertTrue(np.allclose(psnr[1] - psnr[0], 10, atol=0.5))
        self.assertTrue(np.all(np.isnan(analyse(ir, edcMethod='noCut').PSNR)))

    def test_bass_and_treble_ratios(self):
        # Octave bands decaying with T of 2.0, 1.8, 1.5, 1.5, 1.2 and 1.0 s
        # from 125 Hz to 4 kHz: BR = (2.0 + 1.8) / 3.0, TR = (1.2 + 1.0) / 3.0.
        rng = np.random.default_rng(1)
        t = np.arange(4 * FS) / FS
        h = np.zeros(t.size)
        for fc, Tb in zip((125, 250, 500, 1000, 2000, 4000),
                          (2.0, 1.8, 1.5, 1.5, 1.2, 1.0)):
            sos = ss.butter(8, [fc / 2 ** 0.5, fc * 2 ** 0.5], 'bandpass',
                            fs=FS, output='sos')
            h += ss.sosfilt(sos, rng.standard_normal(t.size)) \
                * 10 ** (-3 * t / Tb)
        ra = pytta.RoomAnalysis(pytta.SignalObj(h, 'time', FS,
                                                signalType='energy'),
                                nthOct=1, minFreq=125., maxFreq=4000.)
        BR, TR = ra.timbre_ratios('T30')
        print(f'\n  BR {BR:.3f} (expected {3.8 / 3:.3f}), TR {TR:.3f} '
              f'(expected {2.2 / 3:.3f})')
        self.assertAlmostEqual(BR / (3.8 / 3), 1, delta=0.05)
        self.assertAlmostEqual(TR / (2.2 / 3), 1, delta=0.05)
