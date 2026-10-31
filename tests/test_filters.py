"""
Tests for the filters of pytta.functions that follow the ITA-Toolbox
(ita_butterFR, ita_liriFR, ita_filter_LiRi, ita_filter_notch,
ita_filter_peak). Each test pins one rule of the filter design.

The filters act on the spectrum, so an impulse gives the response itself
at the FFT bins.
"""
import os
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

FS, N = 48000, 48000      # 1 Hz bins


def response(sig):
    return np.fft.rfft(sig.timeSignal[:, 0].astype(float))


def impulse():
    label = 'impulse'
    x = np.zeros(N)
    x[0] = 1
    return pytta.SignalObj(x, 'time', FS)


class TestFilters(unittest.TestCase):

    def test_butterworth_is_3_dB_down_at_the_cutoff(self):
        # |H(f)|^2 = 1 / (1 + (f / fc)^(2n)): 1/2 at fc for every order, and
        # from 2 fc to 4 fc the level falls by 10 lg((1 + 16^n) / (1 + 4^n))
        # (fc = 100 Hz, far below Nyquist, where the bilinear transform bends
        # the response, and above the float32 floor of SignalObj).
        for order in (2, 4, 8):
            H = response(pytta.butterworth(impulse(), 1000, order=order))
            at_fc = 20 * np.log10(np.abs(H[1000]))
            H = response(pytta.butterworth(impulse(), 100, order=order))
            slope = 20 * np.log10(np.abs(H[400]) / np.abs(H[200]))
            print(f'\n  order {order}: {at_fc:.4f} dB at fc, {slope:.2f} dB per octave above', end='')
            self.assertAlmostEqual(at_fc, -3.0103, delta=1e-3)
            self.assertAlmostEqual(slope, -10 * np.log10((1 + 16 ** order) / (1 + 4 ** order)), delta=0.02)

    def test_linkwitz_riley_pair_sums_to_an_allpass(self):
        # A Linkwitz-Riley pair is 6 dB down at the crossover and the low
        # and high branches sum to a flat magnitude.
        lo = response(pytta.butterworth(impulse(), 2000, 'low', 4, linkwitz_riley=True))
        hi = response(pytta.butterworth(impulse(), 2000, 'high', 4, linkwitz_riley=True))
        flat = np.abs(np.abs(lo + hi)[20:20000] - 1).max()
        at_fc = 20 * np.log10(np.abs(lo[2000]))
        print(f'\n  {at_fc:.4f} dB at the crossover, |lo + hi| within {flat:.1e} of 1')
        self.assertAlmostEqual(at_fc, -6.0206, delta=1e-3)
        self.assertLess(flat, 1e-6)

    def test_notch_and_peak(self):
        # The notch is zero at fc and 1 at DC and Nyquist; the peak filter
        # has its gain at fc and 1 at DC and Nyquist.
        nH = np.abs(response(pytta.notch(impulse(), 1000, Q=5)))
        pH = np.abs(response(pytta.peak_eq(impulse(), 3000, 6, Q=2)))
        print(f'\n  notch: {nH[1000]:.1e} at fc, {nH[0]:.6f} at DC; '
              f'peak: {20 * np.log10(pH[3000]):.4f} dB at fc, {pH[0]:.6f} at DC')
        self.assertLess(nH[1000], 1e-6)
        self.assertAlmostEqual(nH[0], 1, places=6)
        self.assertAlmostEqual(20 * np.log10(pH[3000]), 6, delta=1e-4)
        self.assertAlmostEqual(pH[0], 1, places=6)
        self.assertAlmostEqual(pH[-1], 1, places=6)

    def test_zero_phase_and_rejection(self):
        H = response(pytta.butterworth(impulse(), 1000, zero_phase=True))
        self.assertLess(np.abs(H.imag).max(), 1e-6)
        with self.assertRaises(ValueError):
            pytta.butterworth(impulse(), 1000, 'notch')
        with self.assertRaises(ValueError):
            pytta.butterworth(impulse(), 1000, order=3, linkwitz_riley=True)


if __name__ == '__main__':
    unittest.main()
