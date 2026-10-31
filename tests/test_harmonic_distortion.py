"""
Tests for pytta.harmonic_distortion.

An exponential sweep separates the harmonics of a nonlinear system in the
impulse response obtained by deconvolution: the harmonic of order k lands
L ln(k) seconds before the linear response, where L is the time the sweep
takes to multiply its frequency by e (Farina, AES 108th Convention, 2000).

The tests use a memoryless system y = x + a x**2 + b x**3 driven by a sweep
of amplitude A, whose harmonics are known: the linear response has gain
1 + 3 b A**2 / 4, the second harmonic a A / 2 and the third b A**2 / 4,
relative to the sweep. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import scipy.signal as ss

import pytta

FS = 48000
F1, F2, T = 20.0, 7000.0, 2.0     # 3 x F2 stays below Nyquist
A = 0.5
RATE = np.log2(F2 / F1) / T       # octaves per second


def sweep_and_ir(system):
    """Sweep of amplitude A, 0.5 s of silence after it, and the impulse
    response of `system` by regularized circular deconvolution."""
    t = np.arange(int(T * FS)) / FS
    x = np.concatenate((A * ss.chirp(t, F1, T, F2, 'logarithmic', phi=-90),
                        np.zeros(FS // 2)))
    X, Y = np.fft.rfft(x), np.fft.rfft(system(x))
    h = np.fft.irfft(Y * np.conj(X) / (np.abs(X) ** 2 + 1e-8 * np.max(
        np.abs(X) ** 2)), n=x.size)
    return x / A, h


class TestHarmonicDistortion(unittest.TestCase):

    def test_harmonic_levels_of_a_known_nonlinearity(self):
        a, b = 0.1, 0.2
        x, h = sweep_and_ir(lambda x: x + a * x ** 2 + b * x ** 3)
        ir = pytta.SignalObj(h, 'time', FS)
        harmonics, f, thd = pytta.harmonic_distortion(ir, RATE,
                                                      numHarmonics=3)
        band = (f >= 100) & (f <= 2000)
        H = np.abs(np.fft.rfft(harmonics.timeSignal, axis=0))
        i = np.flatnonzero(band)
        gains = [np.median(H[i, 0]), np.median(H[2 * i, 1] / H[i, 0]),
                 np.median(H[3 * i, 2] / H[i, 0])]
        H1 = 1 + 3 * b * A ** 2 / 4
        expected = [H1, a * A / 2 / H1, b * A ** 2 / 4 / H1]
        for k, (got, exp) in enumerate(zip(gains, expected), start=1):
            print(f'\n  harmonic {k}: {got:.5f} (expected {exp:.5f})')
            self.assertAlmostEqual(got / exp, 1, delta=0.02)
        THD = np.hypot(expected[1], expected[2])
        worst = np.max(np.abs(thd[band] / THD - 1))
        print(f'\n  THD {THD:.5f}, worst relative error 100 Hz to 2 kHz '
              f'{worst:.1e}')
        self.assertLess(worst, 0.05)

    def test_linear_system_shows_no_distortion(self):
        # A resonance whose impulse response decays well before the
        # windows of the harmonics: no energy may leak into them.
        b, a = ss.iirpeak(1000, 5, fs=FS)
        x, h = sweep_and_ir(lambda x: ss.lfilter(b, a, x))
        ir = pytta.SignalObj(h, 'time', FS)
        _, f, thd = pytta.harmonic_distortion(ir, RATE, numHarmonics=3)
        band = (f >= 100) & (f <= 2000)
        print(f'\n  largest THD 100 Hz to 2 kHz: {thd[band].max():.1e}')
        self.assertLess(thd[band].max(), 1e-3)

    def test_rate_is_read_from_the_sweep(self):
        # Passing the sweep instead of its rate gives the same result.
        x, h = sweep_and_ir(lambda x: x + 0.1 * x ** 2)
        ir = pytta.SignalObj(h, 'time', FS)
        sweep = pytta.SignalObj(x, 'time', FS, freqMin=F1, freqMax=F2)
        _, _, fromRate = pytta.harmonic_distortion(ir, RATE, 2)
        _, _, fromSweep = pytta.harmonic_distortion(ir, sweep, 2)
        self.assertTrue(np.allclose(fromSweep, fromRate, rtol=1e-3))

    def test_rejects_invalid_arguments(self):
        ir = pytta.SignalObj(np.zeros((FS, 2)), 'time', FS)
        with self.assertRaises(ValueError):
            pytta.harmonic_distortion(ir, RATE)            # two channels
        ir = pytta.SignalObj(np.zeros(FS), 'time', FS)
        with self.assertRaises(ValueError):
            pytta.harmonic_distortion(ir, RATE, numHarmonics=1)
        with self.assertRaises(TypeError):
            pytta.harmonic_distortion(np.zeros(FS), RATE)


if __name__ == '__main__':
    unittest.main()
