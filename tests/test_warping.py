"""
Tests for warp and warp_zpk of pytta.functions, which follow ita_audio_warp
and ita_zpk_warp of the ITA-Toolbox. Each test compares with the analytic
frequency mapping of the allpass D(z) = (z^-1 - lam) / (1 - lam z^-1).
"""
import unittest

import numpy as np
import scipy.signal as ss

import pytta

LAM = 0.6


def theta(w, lam=LAM):
    # Phase of the allpass: D(e^jw) = e^(-j theta(w)).
    return w + 2 * np.arctan2(lam * np.sin(w), 1 - lam * np.cos(w))


class TestWarping(unittest.TestCase):

    def test_warped_signal_has_the_mapped_spectrum(self):
        label = 'x'
        h = np.random.default_rng(5).standard_normal(64) * 0.9 ** np.arange(64)
        y = pytta.warp(pytta.SignalObj(h, 'time', 48000), LAM, 4096).timeSignal[:, 0].astype(float)
        w = np.linspace(0.05, 3, 25)
        Y = np.exp(-1j * np.outer(w, np.arange(y.size))) @ y
        H = np.exp(-1j * np.outer(theta(w), np.arange(h.size))) @ h
        e = np.abs(Y - H).max() / np.abs(H).max()
        print(f'\n  warped spectrum: {e:.1e}', end='')
        self.assertLess(e, 1e-6)

    def test_warped_filter_has_the_mapped_response_and_unwarps(self):
        z, p, k = ss.ellip(4, 1, 40, 0.2, output='zpk')
        zw, pw, kw = pytta.warp_zpk(z, p, k, LAM)
        w = np.linspace(0.05, 3, 25)
        e = np.abs(ss.freqz_zpk(zw, pw, kw, w)[1] - ss.freqz_zpk(z, p, k, theta(w))[1]).max()
        zb, pb, kb = pytta.warp_zpk(zw, pw, kw, -LAM)
        back = np.abs(ss.freqz_zpk(zb, pb, kb, w)[1] - ss.freqz_zpk(z, p, k, w)[1]).max()
        print(f'\n  warped filter: {e:.1e}, unwarped {back:.1e}', end='')
        self.assertLess(max(e, back), 1e-10)


if __name__ == '__main__':
    unittest.main()
