"""
Tests for pytta.bem, collocation with constant triangles for radiation into
free field. Each test compares with the analytic field of a sphere.
"""
import unittest

import numpy as np

from pytta import bem

A, C, RHO = 0.1, 343., 1.21
NODES, TRIS = bem.icosphere(A, 3)
POINTS = np.array([[0, 0, 1.], [0.5, 0.2, 0.1], [-0.3, -0.3, 0.6]])


def pulsating(k, r):
    # rho c U jka / (1 + jka) (a / r) exp(-jk(r - a)) for U = 1 m/s.
    return RHO * C * 1j * k * A / (1 + 1j * k * A) * A / r * np.exp(-1j * k * (r - A))


def oscillating(k, r, cos):
    # Dipole field of a rigid sphere oscillating along z with U = 1 m/s:
    # p = B cos (1 + jkr) exp(-jkr) / r^2, with dp/dr = -j w rho U cos at a.
    w = k * C
    B = -1j * w * RHO * A**3 * np.exp(1j * k * A) / (k**2 * A**2 - 2j * k * A - 2)
    return B * cos * (1 + 1j * k * r) * np.exp(-1j * k * r) / r**2


class TestBEM(unittest.TestCase):

    def test_pulsating_sphere_and_chief(self):
        # Surface and field within 1 %; at ka = pi, an irregular frequency
        # of the surface equation, the CHIEF points are what keeps it so.
        r = np.linalg.norm(POINTS, axis=1)
        for ka in (0.55, 1.83, np.pi):
            k = ka / A
            f = k * C / (2 * np.pi)
            p = bem.surface_pressure(NODES, TRIS, f, 1.)
            pf = bem.field(NODES, TRIS, f, p, 1., POINTS)
            es = np.abs(p / pulsating(k, A) - 1).max()
            ef = np.abs(pf / pulsating(k, r) - 1).max()
            print(f'\n  pulsating sphere, ka {ka:.2f}: surface {es:.1e}, field {ef:.1e}', end='')
            self.assertLess(max(es, ef), 0.01)
        bare = bem.surface_pressure(NODES, TRIS, np.pi / A * C / (2 * np.pi), 1., chief=np.zeros((0, 3)))
        self.assertGreater(np.abs(bare / pulsating(np.pi / A, A) - 1).max(), 0.05)

    def test_oscillating_sphere(self):
        k = 1.5 / A
        f = k * C / (2 * np.pi)
        P, centre, normal, _ = bem._geometry(NODES, TRIS)
        u = normal[:, 2]                          # U cos(theta) along z
        p = bem.surface_pressure(NODES, TRIS, f, u)
        pf = bem.field(NODES, TRIS, f, p, u, POINTS)
        r = np.linalg.norm(POINTS, axis=1)
        exact = oscillating(k, r, POINTS[:, 2] / r)
        ef = np.abs(pf - exact).max() / np.abs(exact).max()
        es = np.abs(p - oscillating(k, A, centre[:, 2] / A)).max() / np.abs(oscillating(k, A, 1.))
        print(f'\n  oscillating sphere, ka 1.5: surface {es:.1e}, field {ef:.1e}', end='')
        self.assertLess(max(es, ef), 0.02)


if __name__ == '__main__':
    unittest.main()
