"""
Tests for prony and invfreqz of pytta.functions, the time and frequency
domains of ita_prony_analysis of the ITA-Toolbox. Each test recovers the
coefficients of a known filter.
"""
import unittest

import numpy as np
import scipy.signal as ss

import pytta


class TestProny(unittest.TestCase):

    def test_prony_recovers_an_iir_filter(self):
        # Impulse response of a filter with 3 zeros and 5 poles.
        a = ss.cheby1(5, 1, 0.3)[1]
        b = np.poly([0.9, -0.5, 0.2])
        h = ss.lfilter(b, a, np.r_[1, np.zeros(199)])
        bp, ap = pytta.prony(h, 3, 5)
        e = max(np.abs(bp - b).max() / np.abs(b).max(), np.abs(ap - a).max() / np.abs(a).max())
        print(f'\n  prony: {e:.1e}', end='')
        self.assertLess(e, 1e-9)

    def test_invfreqz_recovers_an_iir_filter(self):
        b, a = ss.butter(4, 0.3)
        w, H = ss.freqz(b, a, 512)
        bi, ai = pytta.invfreqz(H, w, 4, 4)
        e = max(np.abs(bi - b).max() / np.abs(b).max(), np.abs(ai - a).max())
        print(f'\n  invfreqz: {e:.1e}', end='')
        self.assertLess(e, 1e-9)


if __name__ == '__main__':
    unittest.main()
