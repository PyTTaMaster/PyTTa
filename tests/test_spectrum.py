"""
Tests for the spectrum utilities of pytta.functions that follow the
ITA-Toolbox (ita_psd, ita_interpolate_spk, ita_spk_extrapolate). Each test
pins one rule.

The golden data come from golden/make_psd_matlab.m: MATLAB cpsd called as
ita_psd calls it; the noise is rebuilt with numpy.random.RandomState.
"""
import os
import unittest

import numpy as np
import scipy.io as sio
import scipy.signal as ss

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


class TestSpectrum(unittest.TestCase):

    def test_psd_matches_matlab_cpsd_and_parseval(self):
        label = 'psd'
        d = sio.loadmat(os.path.join(GOLDEN, 'psd_matlab.mat'), squeeze_me=True)
        fs, block = int(d['fs']), int(d['block'])
        x = np.random.RandomState(1).random_sample(4096) - 0.5
        y = ss.lfilter([1, 0.5, 0.25], 1, x)
        sx, sy = (pytta.SignalObj(v, 'time', fs) for v in (x, y))
        # SignalObj stores float32: compare with the float32 samples
        x32, y32 = sx.timeSignal[:, 0].astype(float), sy.timeSignal[:, 0].astype(float)
        f, Pxx = pytta.psd(sx, block_size=block)
        _, Pxy = pytta.psd(sx, sy, block_size=block)
        e_xx = np.abs(Pxx[:, 0] / d['Pxx'] - 1).max()
        e_xy = np.abs(Pxy[:, 0] / d['Pxy'] - 1).max()
        power = np.sum(Pxx[:, 0]) * (f[1] - f[0])
        print(f'\n  PSD {e_xx:.1e}, CPSD {e_xy:.1e} relative to MATLAB; integral {power:.5f} '
              f'against the variance {np.var(x32):.5f}')
        self.assertLess(e_xx, 1e-5)
        self.assertLess(e_xy, 1e-5)
        self.assertAlmostEqual(power / np.var(x32), 1, delta=0.02)

    def test_interpolated_spectrum_follows_the_true_response(self):
        # h[n] = a^n has H = 1 / (1 - a exp(-j w)); its 256-point DFT,
        # interpolated onto the bins of 1024 points, follows H there.
        label = 'interp'
        fs, a = 48000, 0.5
        h = a ** np.arange(256)
        sig = pytta.SignalObj(h, 'time', fs)
        fine = pytta.interpolate_spectrum(sig, 1024)
        H = np.fft.rfft(fine.timeSignal[:, 0].astype(float))
        w = 2 * np.pi * np.arange(H.size) / 1024
        err = np.abs(H - 1 / (1 - a * np.exp(-1j * w))).max()
        same = pytta.interpolate_spectrum(sig, 256).timeSignal[:, 0]
        print(f'\n  max |H - H_true| on the finer grid {err:.1e}; same length {np.abs(same - h).max():.1e}')
        self.assertLess(err, 1e-3)
        self.assertLess(np.abs(same - h).max(), 1e-6)

    def test_low_extrapolation(self):
        # A pure delay has a flat magnitude and the group delay of the
        # delay, so its extrapolation reproduces it; a slope of 6 dB per
        # decade leaves the magnitude 6 dB lower a decade below the range.
        label = 'extrap'
        fs, n, k = 1000, 1000, 25
        x = np.zeros(n)
        x[k] = 1
        sig = pytta.SignalObj(x, 'time', fs)
        same = pytta.extrapolate_low(sig, (100, 200)).timeSignal[:, 0]
        sloped = np.abs(np.fft.rfft(pytta.extrapolate_low(sig, (100, 200), slope=6).timeSignal[:, 0]))
        ratio = 20 * np.log10(sloped[10] / sloped[100])
        print(f'\n  pure delay kept within {np.abs(same - x).max():.1e}; '
              f'{ratio:.4f} dB from 100 Hz to 10 Hz')
        self.assertLess(np.abs(same - x).max(), 1e-6)
        self.assertAlmostEqual(ratio, -6, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
