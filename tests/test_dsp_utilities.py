"""
Tests for the DSP utilities group_delay, coherence, cepstrum, icepstrum,
generate.mls and mls_deconvolution.

Each test pins one rule with an analytic value. SignalObj stores its samples
in float32, so the exact cases are checked to about 1e-6.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import scipy.signal as ss

import pytta

FS = 48000
TOL32 = 1e-6  # rounding of the float32 SignalObj storage


class TestGroupDelay(unittest.TestCase):

    def test_pure_delay_and_first_order_allpass(self):
        # A delay of k samples has group delay k / fs at every frequency.
        # The allpass H(z) = (a + z^-1) / (1 + a z^-1) has group delay
        # (1 - a^2) / (1 + 2 a cos(w) + a^2) samples.
        N, k, a = 1024, 37, 0.8
        data = np.zeros((N, 2))
        data[k, 0] = 1
        impulse = np.zeros(N)
        impulse[0] = 1
        data[:, 1] = ss.lfilter([a, 1], [1, a], impulse)
        sig = pytta.SignalObj(data, 'time', FS)
        tau = pytta.group_delay(sig)
        w = 2 * np.pi * sig.freqVector / FS
        expected = np.column_stack([np.full(w.size, k / FS),
                                    (1 - a**2) / (1 + 2*a*np.cos(w) + a**2) / FS])
        err = np.max(np.abs(tau - expected), axis=0) * FS
        print(f'\n  max error: delay {err[0]:.1e}, allpass {err[1]:.1e} samples')
        self.assertEqual(tau.shape, (N // 2 + 1, 2))
        self.assertLess(np.max(err), 1e-6)


class TestCoherence(unittest.TestCase):

    def test_noisy_output_and_identical_signals(self):
        # y = h * x + n with white x (variance 1) and white n (variance s2):
        # SNR(f) = |H(f)|^2 / s2 and gamma^2 = SNR / (1 + SNR). With 8192
        # Welch segments the bias is below 1e-3 and the random error has a
        # standard deviation of about 0.004, so 0.02 is five of them.
        # A signal against itself has gamma^2 = 1.
        rng = np.random.default_rng(0)
        N, s2, h = 2**19, 0.5, [1, 0.9]
        x = rng.standard_normal(N)
        y = ss.lfilter(h, 1, x) + np.sqrt(s2) * rng.standard_normal(N)
        sigX = pytta.SignalObj(x, 'time', FS)
        sigY = pytta.SignalObj(np.column_stack([y, x]), 'time', FS)
        freq, gamma2 = pytta.coherence(sigX, sigY, winSize=128)
        H2 = np.abs(ss.freqz(h, 1, worN=freq, fs=FS)[1])**2
        snr = H2 / s2
        err = np.max(np.abs(gamma2[:, 0] - snr / (1 + snr)))
        errSelf = np.max(np.abs(gamma2[:, 1] - 1))
        print(f'\n  max |error|: noisy {err:.1e}, identical {errSelf:.1e}')
        self.assertLess(err, 0.02)
        self.assertLess(errSelf, TOL32)


class TestCepstrum(unittest.TestCase):

    def test_known_minimum_phase_filter_matches_analytic_series(self):
        # For x = (1 - a z^-1)(1 - b z^-1) with |a|, |b| < 1 the complex
        # cepstrum is -(a^n + b^n) / n for n > 0 and zero for n <= 0
        # (Oppenheim and Schafer, DTSP 3rd ed., eq. 13.36). The real
        # cepstrum is its even part.
        N, a, b = 1024, 0.9, -0.6
        x = np.zeros(N)
        x[:3] = np.poly([a, b])
        sig = pytta.SignalObj(x, 'time', FS)
        n = np.arange(1, N // 2)
        series = np.zeros(N)
        series[n] = -(a**n + b**n) / n
        real = pytta.cepstrum(sig, kind='real').timeSignal[:, 0]
        cplx, delay = pytta.cepstrum(sig, kind='complex')
        errReal = np.max(np.abs(real - (series + np.roll(series[::-1], 1)) / 2))
        errCplx = np.max(np.abs(cplx.timeSignal[:, 0] - series))
        print(f'\n  max |error|: real {errReal:.1e}, complex {errCplx:.1e}')
        self.assertLess(errReal, TOL32)
        self.assertLess(errCplx, TOL32)
        self.assertEqual(list(delay), [0])

    def test_complex_cepstrum_round_trip_with_linear_phase(self):
        # A mixed phase filter (one zero outside the unit circle) delayed by
        # 7 samples: the linear phase removed is 7 samples of delay plus one
        # per zero outside the circle, and icepstrum puts it back.
        N = 1024
        x = np.zeros((N, 2))
        x[7:10, 0] = np.poly([0.8, -1.6])
        x[:, 1] = np.random.default_rng(1).standard_normal(N) \
            * np.exp(-np.arange(N) / 50)
        x[:, 1] *= np.sign(x[:, 1].sum())  # positive gain at 0 Hz
        sig = pytta.SignalObj(x, 'time', FS)
        cep, delay = pytta.cepstrum(sig)
        back = pytta.icepstrum(cep, delay)
        err = np.max(np.abs(back.timeSignal - sig.timeSignal), axis=0)
        print(f'\n  delay {delay}, max |error| {err[0]:.1e}, {err[1]:.1e}')
        self.assertEqual(delay[0], 8)
        self.assertLess(np.max(err), TOL32)


class TestMLS(unittest.TestCase):

    def test_periodic_autocorrelation_is_exact(self):
        # A symmetrical MLS of period L = 2^N - 1 has periodic
        # autocorrelation (L + 1) delta - 1 (Peltonen 2000, eq. 2.14-2.15).
        order = 10
        L = 2**order - 1
        s = pytta.generate.mls(order, samplingRate=FS)
        z = s.timeSignal[:, 0].astype(int)
        R = np.array([np.dot(z, np.roll(z, k)) for k in range(L)])
        expected = (L + 1) * (np.arange(L) == 0) - 1
        print(f'\n  L = {s.numSamples}, max |R - expected| = '
              f'{np.max(np.abs(R - expected))}')
        self.assertEqual(s.numSamples, L)
        self.assertTrue(np.array_equal(R, expected))

    def test_deconvolution_of_fir_excited_by_periodic_mls(self):
        # Periodic cross-correlation normalized by L + 1 returns
        # h - sum(h) / (L + 1) (Peltonen 2000, eq. 2.22): the FIR within
        # 1 / L, with that exact constant offset.
        order = 12
        L = 2**order - 1
        h = np.random.default_rng(2).standard_normal(40) \
            * np.exp(-np.arange(40) / 8)
        z = pytta.generate.mls(order).timeSignal[:, 0]
        y = ss.lfilter(h, 1, np.tile(z, 3))
        resp = pytta.SignalObj(np.column_stack([y, 2 * y]), 'time', FS)
        ir = pytta.mls_deconvolution(resp, order)
        hL = np.zeros(L)
        hL[:h.size] = h
        expected = hL - h.sum() / (L + 1)
        errOffset = np.max(np.abs(ir.timeSignal[:, 0] - expected))
        errFir = np.max(np.abs(ir.timeSignal[:, 0] - hL))
        err2 = np.max(np.abs(ir.timeSignal[:, 1] - 2 * expected))
        print(f'\n  max |ir - h| = {errFir:.1e} (1/L = {1/L:.1e}), '
              f'with offset {errOffset:.1e}, channel 2 {err2:.1e}')
        self.assertEqual(ir.numSamples, L)
        self.assertLess(errFir, 1 / L)
        self.assertLess(max(errOffset, err2), TOL32)


if __name__ == '__main__':
    unittest.main()
