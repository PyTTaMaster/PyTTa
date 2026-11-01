"""
Tests for power_to_harmonic_series and harmonic_to_power_series of
pytta.functions, which follow ita_nonlinear_power_to_harmonic_series and
ita_nonlinear_harmonic_to_power_series of the ITA-Toolbox and Dietrich,
Uncertainties in acoustical transfer functions (2013), sec. 3.3.
"""
import unittest

import numpy as np

import pytta


class TestNonlinearSeries(unittest.TestCase):

    def test_example_of_dietrich(self):
        # y = x + x^2 + x^3 + x^4 driven with G = 1/2 (Dietrich 2013, p. 51).
        h = pytta.power_to_harmonic_series([1, 1, 1, 1], gain=0.5)
        print(f'\n  harmonics {h}', end='')
        np.testing.assert_allclose(h, [1.1875, 0.3125, 0.0625, 0.0156], atol=5e-5)

    def test_harmonics_of_a_driven_polynomial(self):
        # The DFT of the polynomial of gain cos(wt), over one period and
        # divided by gain, has the predicted harmonics.
        a = np.random.default_rng(1).standard_normal(6)
        g, n = 0.8, 64
        x = g * np.cos(2 * np.pi * np.arange(n) / n)
        y = sum(a[k] * x ** (k + 1) for k in range(6))
        dft = 2 * np.fft.rfft(y)[1:7].real / n / g
        e = np.abs(pytta.power_to_harmonic_series(a, gain=g) - dft).max()
        print(f'\n  against the DFT: {e:.1e}', end='')
        self.assertLess(e, 1e-13)

    def test_phase_and_inverse(self):
        # Harmonic m turns by (m - 1) phase, as in the ITA-Toolbox, and the
        # inverse gives back coefficients per frequency.
        h = pytta.power_to_harmonic_series([1, 1, 1, 1], phase=0.3)
        np.testing.assert_allclose(np.angle(h), [0, 0.3, 0.6, 0.9])
        a = np.random.default_rng(2).standard_normal((5, 100)) + 1j
        back = pytta.harmonic_to_power_series(pytta.power_to_harmonic_series(a, 0.3, 1.1), 0.3, 1.1)
        print(f'\n  inverse: {np.abs(back - a).max():.1e}', end='')
        self.assertLess(np.abs(back - a).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
