"""
Tests for the energy parameters of pytta.RoomAnalysis computed by windowing
the broadband impulse response before the band filtering (ISO 3382-1:2009,
A.3.4). The responses are synthetic, so the reference values are known.
"""
import os
import unittest

import numpy as np
import scipy.signal as ss

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta
from pytta.classes.analysis import clarity

FS = 48000
BANDS = dict(nthOct=1, minFreq=125, maxFreq=4000)


def decay(seed, T=1.0, seconds=1.5):
    # Exponentially decaying white noise: the energy falls by 60 dB in T.
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = np.exp(-6.9078 * t / T) * rng.standard_normal(t.size)
    return 0.5 * x / np.abs(x).max()


def room(*samples, **kwargs):
    # A tuple as first local variable: SignalObj compares it with itself
    # while walking the stack, which an array cannot do before #59.
    ir = pytta.SignalObj(samples[0], 'time', FS)
    return pytta.RoomAnalysis(ir, suppressWarnings=True, **BANDS, **kwargs)


class TestRoomEnergyWindowing(unittest.TestCase):

    def test_c80_follows_iso_3382_1_a34(self):
        # The early 80 ms and the rest of the cropped broadband response,
        # filtered separately with the same band filters (their sos from
        # OctFilter, applied with scipy) and summed over the whole output.
        filters = pytta.OctFilter(order=4, samplingRate=FS, refFreq=1000, base=10, **BANDS)
        k = int(0.08 * FS)
        for seed in (1, 2, 3):
            r = room(decay(seed), bypassLundeby=True)
            x = np.r_[np.asarray(r.ir.timeSignal[:, 0], dtype=float), np.zeros(FS)]
            early, late = x.copy(), x.copy()
            early[k:], late[:k] = 0, 0
            ref = [10 * np.log10(np.sum(ss.sosfilt(np.ascontiguousarray(filters.sos[:, :, b]), early)**2)
                                 / np.sum(ss.sosfilt(np.ascontiguousarray(filters.sos[:, :, b]), late)**2))
                   for b in range(filters.sos.shape[2])]
            e = np.abs(r.C80 - ref).max()
            print(f'\n  C80 against A.3.4, response {seed}: {e:.3f} dB', end='')
            self.assertLess(e, 0.011)        # C80 is rounded to 0.01 dB

    def test_no_bias_against_the_exponential_decay(self):
        # For T = 1 s the expected values are C80 = 10 lg(e^(13.8155 0.08)
        # - 1) = 3.05 dB and D50 = 1 - e^(-13.8155 0.05) = 49.88 %. Over 20
        # responses the means stay within three standard errors in every
        # band, where the energy decay curve of the filtered response, used
        # before, is low by about 1.4 dB at 125 Hz.
        new, old, d50 = [], [], []
        for seed in range(20):
            r = room(decay(100 + seed))
            new.append(r.C80)
            d50.append(r.D50)
            old.append(clarity(r.listEDC, FS))
        a = 13.8155
        for name, values, expected in (('C80 windowed', new, 10 * np.log10(np.exp(a * 0.08) - 1)),
                                       ('D50 windowed', d50, 100 * (1 - np.exp(-a * 0.05))),
                                       ('C80 from the EDC', old, 10 * np.log10(np.exp(a * 0.08) - 1))):
            v = np.array(values)
            z = (v.mean(0) - expected) / (v.std(0, ddof=1) / np.sqrt(len(v)))
            print(f'\n  {name}: mean - expected {np.round(v.mean(0) - expected, 2)}, '
                  f'in standard errors {np.round(z, 1)}', end='')
            if name.endswith('windowed'):
                self.assertTrue(np.all(np.abs(z) < 3))
            else:
                self.assertLess(z[0], -3)

    def test_stlate_ends_at_one_second(self):
        # Impulses of energy 1 at the start, 0.25 0.5 s later and 1 1.2 s
        # later: STlate = 10 lg(0.25 / 1) = -6.02 dB in every band, since the
        # last impulse lies outside its window (ISO 3382-1, C.2). Silence
        # before the first impulse lets the start detection find it.
        x = np.zeros(int(1.5 * FS))
        n0 = int(0.01 * FS)
        x[[n0, n0 + int(0.5 * FS), n0 + int(1.2 * FS)]] = 1, 0.5, 1
        r = room(x, bypassLundeby=True, ircut=1.45)
        print(f'\n  STlate {np.round(r.STlate, 3)}', end='')
        np.testing.assert_allclose(r.STlate, 10 * np.log10(0.25), atol=0.01)


if __name__ == '__main__':
    unittest.main()
