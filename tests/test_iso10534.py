"""
Tests for pytta.iso10534, the two-microphone transfer function method of
ISO 10534-2 for impedance tubes. The transfer functions are synthesized
with Equation (D.7) for known reflection factors, so the expected values
are exact. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta
from pytta import iso10534

F = np.linspace(100, 1800, 200)
C = iso10534.speed_of_sound(20.0)
X1, S = 0.10, 0.05                       # m: sample to far microphone, spacing


def h12(r):
    """Equation (D.7): H12 = p2 / p1 for a reflection factor r at x = 0."""
    k = 2 * np.pi * F / C
    x2 = X1 - S
    return (np.exp(1j * k * x2) + r * np.exp(-1j * k * x2)) \
        / (np.exp(1j * k * X1) + r * np.exp(-1j * k * X1))


class TestISO10534(unittest.TestCase):

    def test_speed_of_sound(self):
        # Equation (5): c0 = 343.2 sqrt(T / 293), T in kelvin.
        self.assertAlmostEqual(iso10534.speed_of_sound(19.85), 343.2, 6)

    def test_known_reflection_factors_come_back(self):
        # Rigid termination (r = 1, alpha = 0), anechoic (r = 0, alpha = 1,
        # Z = rho c) and a complex r with frequency-dependent phase.
        cases = (np.ones(F.size), np.zeros(F.size),
                 0.5 * np.exp(1j * (np.pi / 4 + F / 1000)))
        for r in cases:
            got = iso10534.reflection_factor(h12(r), F, X1, S, C)
            self.assertTrue(np.allclose(got, r, atol=1e-9))
            alpha = iso10534.absorption(got)
            self.assertTrue(np.allclose(alpha, 1 - np.abs(r) ** 2, atol=1e-9))
        z = iso10534.impedance_ratio(np.zeros(F.size))
        self.assertTrue(np.allclose(z, 1))

    def test_transfer_function_from_two_responses(self):
        # p2 is p1 delayed by d samples and halved: H12 = 0.5 exp(-j w d).
        fs, d = 48000, 7
        p1 = pytta.SignalObj(np.eye(1, 4096, 100)[0], 'time', fs)
        p2 = pytta.SignalObj(0.5 * np.eye(1, 4096, 100 + d)[0], 'time', fs)
        f, H = iso10534.transfer_function(p1, p2)
        self.assertTrue(np.allclose(H, 0.5 * np.exp(-2j * np.pi * f * d / fs)))

    def test_working_frequency_range(self):
        # Equations (2) and (4) and the 5 % rule: a 100 mm circular tube
        # with microphones 50 mm apart.
        c = 343.2
        low, high = iso10534.frequency_range(0.1, 0.05, c)
        self.assertAlmostEqual(high, min(0.58 * c / 0.1, 0.45 * c / 0.05))
        self.assertAlmostEqual(low, 0.05 * c / 0.05)


if __name__ == '__main__':
    unittest.main()
