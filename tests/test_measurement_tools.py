"""
Tests for dither, impulsive_noise, invert_spectrum and merge_near_far of
pytta.functions, which cover ita_dither, ita_impulsiveNoiseDetection,
ita_invert_spk_minphase_regularization and
ita_add_nearfield_farfield_measurements of the ITA-Toolbox. Each test pins
one rule on a synthetic measurement.
"""
import unittest

import numpy as np
import scipy.signal as ss

import pytta

FS = 48000


def signal(*samples):
    # A tuple as first local variable: SignalObj compares it with itself
    # while walking the stack, which an array cannot do before #59.
    return pytta.SignalObj(samples[0], 'time', FS)


class TestMeasurementTools(unittest.TestCase):

    def test_triangular_dither_makes_the_error_independent(self):
        # With triangular dither the total error of an 8-bit quantizer has
        # the power step^2 / 4 and no mean, whatever the signal (Lipshitz
        # et al. 1992); without it the error follows the signal.
        label = 'x'
        step = 2.0 ** -7
        x = 1.5 * step * np.sin(2 * np.pi * 50 * np.arange(FS) / FS)
        y = pytta.dither(signal(x), 8, 'tri', seed=1).timeSignal[:, 0].astype(float)
        e = np.round(y / step) * step - x
        power = np.mean(e ** 2) / (step ** 2 / 4)
        bias = max(abs(np.mean(e[x > step])), abs(np.mean(e[x < -step]))) / step
        print(f'\n  dithered error: power {power:.3f} of step^2/4, mean {bias:.1e} step', end='')
        self.assertLess(abs(power - 1), 0.02)
        self.assertLess(bias, 0.01)

    def test_impulsive_noise_is_detected(self):
        # Excitation of flat magnitude and random phase, so that the
        # deconvolution is exact. A click in the recording raises L_max_rms
        # of the rebuilt noise above 20 dB; Gaussian noise alone stays near
        # 13 dB.
        label = 'x'
        rng = np.random.default_rng(2)
        n = 2 ** 16
        X = np.exp(2j * np.pi * rng.uniform(size=n // 2 + 1))
        X[[0, -1]] = 1
        x = np.fft.irfft(X, n)
        room = np.exp(-np.arange(4800) / 600) * rng.standard_normal(4800)
        levels = []
        for click in (0, 0.05):
            y = np.fft.irfft(np.fft.rfft(room, n) * X, n) + 1e-3 * rng.standard_normal(n)
            y[30000] += click
            h = np.fft.irfft(np.fft.rfft(y) / X, n)
            found, level, _ = pytta.impulsive_noise(signal(h), signal(x), 0.2)
            levels.append(level)
            self.assertEqual(found, click > 0)
        print(f'\n  L_max_rms: {levels[0]:.1f} dB without, {levels[1]:.1f} dB with a click', end='')

    def test_inverse_is_exact_in_band_and_minimum_phase(self):
        label = 'x'
        b, a = ss.butter(2, [100, 8000], 'bandpass', fs=FS)
        h = ss.lfilter(b, a, np.r_[1, np.zeros(8191)])
        H = np.fft.rfft(h)
        f = np.fft.rfftfreq(h.size, 1 / FS)
        band = (f > 200) & (f < 4000)
        for minimum in (False, True):
            c = pytta.invert_spectrum(signal(h), [200, 4000], minimum).timeSignal[:, 0].astype(float)
            C = np.fft.rfft(c)
            e = np.abs(np.abs(C * H)[band] - 1).max()
            print(f'\n  inverse, minimum phase {minimum}: in band {e:.1e}', end='')
            self.assertLess(e, 1e-4)
        # The minimum phase inverse has the most energy at its start among
        # inverses with the same magnitude (here the zero phase one).
        zero = np.roll(np.fft.irfft(np.abs(C), h.size), h.size // 2)
        lead = np.cumsum(c ** 2) - np.cumsum(zero ** 2)
        self.assertGreater(lead.min(), -1e-5 * np.sum(c ** 2))   # float32 samples

    def test_near_and_far_field_merge_back(self):
        # A near-field measurement 20 dB louder and 100 samples earlier than
        # the far-field one is aligned, scaled and crossfaded back into it.
        label = 'x'
        h = ss.lfilter(*ss.butter(2, [40, 15000], 'bandpass', fs=FS), np.r_[1, np.zeros(16383)])
        far = np.roll(h, 100)
        merged = pytta.merge_near_far(signal(10 * h), signal(far), 300).timeSignal[:, 0].astype(float)
        e = np.abs(merged - far).max() / np.abs(far).max()
        print(f'\n  near and far field merge: {e:.1e}', end='')
        self.assertLess(e, 1e-4)


if __name__ == '__main__':
    unittest.main()
