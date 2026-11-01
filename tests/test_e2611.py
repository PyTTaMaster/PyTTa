"""
Tests for pytta.e2611, the four-microphone transfer matrix method of ASTM
E2611-09. The microphone pressures are synthesized from a known specimen
matrix with Eq 21 and the plane wave field of the tube, and each test
compares a result with its analytic value.
"""
import unittest

import numpy as np

from pytta import e2611

F = np.linspace(100, 3000, 59)
C, RHO = 343., 1.21
K = 2 * np.pi * F / C
L1, S1, S2 = 0.15, 0.05, 0.05


def microphones(T, d, reflection):
    # Downstream waves C (forward) and D = reflection C, then Eq 21 and
    # T to the front face, and the plane wave field at the microphones.
    l2 = d + 0.1
    Cw = np.ones_like(F, dtype=complex)
    Dw = reflection * Cw
    pd = Cw * np.exp(-1j * K * d) + Dw * np.exp(1j * K * d)
    ud = (Cw * np.exp(-1j * K * d) - Dw * np.exp(1j * K * d)) / (RHO * C)
    p0 = T[:, 0, 0] * pd + T[:, 0, 1] * ud
    u0 = T[:, 1, 0] * pd + T[:, 1, 1] * ud
    A, B = (p0 + RHO * C * u0) / 2, (p0 - RHO * C * u0) / 2
    up = lambda x: A * np.exp(-1j * K * x) + B * np.exp(1j * K * x)
    down = lambda x: Cw * np.exp(-1j * K * x) + Dw * np.exp(1j * K * x)
    return np.array([up(-(L1 + S1)), up(-L1), down(l2), down(l2 + S2)]), l2


def mass_layer(m):
    T = np.zeros((F.size, 2, 2), dtype=complex)
    T[:, 0, 0] = T[:, 1, 1] = 1
    T[:, 0, 1] = 2j * np.pi * F * m
    return T


class TestE2611(unittest.TestCase):

    def test_mass_law_with_two_loads_and_one_load(self):
        # A limp layer of 2 kg/m2 has TL = 20 lg|1 + j w m / (2 rho c)|.
        T = mass_layer(2.)
        Ha, l2 = microphones(T, 0., 0.)
        Hb, _ = microphones(T, 0., 0.9 * np.exp(1j * K * 0.3))
        expected = 20 * np.log10(np.abs(1 + 2j * np.pi * F * 2. / (2 * RHO * C)))
        for name, Hsecond in (('two loads', Hb), ('one load', None)):
            Tm = e2611.transfer_matrix(F, Ha, Hsecond, L1, S1, l2, S2, 0., C, RHO)
            TL = e2611.transmission_loss(Tm, F, 0., C, RHO)[0]
            e = np.abs(TL - expected).max()
            print(f'\n  mass law, {name}: {e:.1e} dB', end='')
            self.assertLess(e, 1e-9)

    def test_air_layer(self):
        # A layer of air 5 cm thick: k' = k, z = rho c, no loss, and with
        # a rigid backing no absorption.
        d = 0.05
        T = np.zeros((F.size, 2, 2), dtype=complex)
        T[:, 0, 0] = T[:, 1, 1] = np.cos(K * d)
        T[:, 0, 1] = 1j * RHO * C * np.sin(K * d)
        T[:, 1, 0] = 1j * np.sin(K * d) / (RHO * C)
        Ha, l2 = microphones(T, d, 0.)
        Hb, _ = microphones(T, d, -0.7)
        Tm = e2611.transfer_matrix(F, Ha, Hb, L1, S1, l2, S2, d, C, RHO)
        k, z = e2611.material_properties(Tm, d)
        TL = e2611.transmission_loss(Tm, F, d, C, RHO)[0]
        alpha = e2611.absorption_coefficient(Tm, C, RHO)
        print(f"\n  air layer: k' {np.abs(k / K - 1).max():.1e}, z {np.abs(z / (RHO * C) - 1).max():.1e}, "
              f'TL {np.abs(TL).max():.1e} dB, alpha {np.abs(alpha).max():.1e}', end='')
        self.assertLess(np.abs(k / K - 1).max(), 1e-9)
        self.assertLess(np.abs(z / (RHO * C) - 1).max(), 1e-9)
        self.assertLess(np.abs(TL).max(), 1e-9)
        self.assertLess(np.abs(alpha).max(), 1e-9)

    def test_air_constants(self):
        # Eq 4 and Eq 5 at 20 degrees Celsius and 101.325 kPa.
        self.assertAlmostEqual(e2611.speed_of_sound(20), 343.24, places=2)
        self.assertAlmostEqual(e2611.air_density(20), 1.2020, places=4)


if __name__ == '__main__':
    unittest.main()
