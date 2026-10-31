"""
Tests for pytta.vibration: frequency weightings of human vibration
(ISO 2631-1, ISO 5349-1, parameters of ISO 8041), the weighted rms, MTVV
and VDV, and integration and differentiation. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import scipy.signal as ss

import pytta
from pytta import vibration

# Weightings in dB at nominal frequencies, computed from the ISO 8041
# parameters (tabulated to 0.01 dB in the Virtins note "Frequency
# weightings for sound and vibration perceived by humans").
TABLE = {'Wk': {0.1: -30.11, 1: -6.33, 4: -0.29, 8: 0.31, 16: -2.28,
                63: -14.61},
         'Wd': {0.1: -24.09, 1: 0.10, 2: -1.01, 8: -11.93, 31.5: -23.98},
         'Wh': {8: -1.18, 16: -0.96, 100: -15.91, 1000: -37.42}}


class TestVibration(unittest.TestCase):

    def test_weightings(self):
        for kind, values in TABLE.items():
            fs = 48000 if kind == 'Wh' else 2000
            sos = vibration.weighting_filter(kind, fs)
            for f, expected in values.items():
                _, h = ss.sosfreqz(sos, worN=np.array([float(f)]), fs=fs)
                self.assertAlmostEqual(20 * np.log10(abs(h[0])), expected,
                                       delta=0.05, msg=(kind, f))
        with self.assertRaises(ValueError):
            vibration.weighting_filter('Wz', 2000)

    def test_rms_mtvv_and_vdv_of_a_sine(self):
        # Unweighted sine of amplitude A over T seconds: rms A / sqrt(2),
        # MTVV the same once settled, VDV A (3 T / 8)^(1/4).
        fs, A, T = 2000, 2.0, 20.0
        t = np.arange(int(T * fs)) / fs
        sig = pytta.SignalObj(A * np.sin(2 * np.pi * 10 * t), 'time', fs)
        self.assertAlmostEqual(vibration.rms(sig)[0], A / np.sqrt(2), 4)
        self.assertAlmostEqual(vibration.mtvv(sig)[0], A / np.sqrt(2),
                               delta=0.01)
        self.assertAlmostEqual(vibration.vdv(sig)[0],
                               A * (3 * T / 8) ** 0.25, 4)

    def test_integration_and_differentiation_of_a_sine(self):
        # A whole number of cycles of sin(w t): the integral is
        # -cos(w t) / w and the derivative w cos(w t).
        fs = 1000
        t = np.arange(fs) / fs
        w = 2 * np.pi * 50
        sig = pytta.SignalObj(np.sin(w * t), 'time', fs)
        integral = vibration.integrate(sig).timeSignal[:, 0]
        derivative = vibration.differentiate(sig).timeSignal[:, 0]
        self.assertTrue(np.allclose(integral, -np.cos(w * t) / w, atol=1e-6))
        self.assertTrue(np.allclose(derivative / w, np.cos(w * t), atol=1e-5))
        twice = vibration.integrate(sig, times=2).timeSignal[:, 0]
        self.assertTrue(np.allclose(twice, -np.sin(w * t) / w ** 2,
                                    atol=1e-8))


if __name__ == '__main__':
    unittest.main()
