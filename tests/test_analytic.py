"""
Tests for pytta.analytic, after the Monopole tools of the ITA-Toolbox. Each
test pins one rule of the field: its form, a boundary condition or a limit.
"""
import unittest

import numpy as np

from pytta import analytic as A

C, RHO = 343.2, 1.204


class TestAnalytic(unittest.TestCase):

    def test_free_field_green_and_plane_wave(self):
        # |G| = 1 / (4 pi r) with phase -k r, the same both ways; a plane
        # wave has unit magnitude and phase -k u.r.
        f = np.array([100., 1000.])
        src, pts = (0.3, -0.2, 1.0), np.array([[1.0, 2.0, 0.5], [-2.0, 0.5, 3.0]])
        G = A.green(f, src, pts)
        r = np.linalg.norm(pts - src, axis=1)
        k = 2 * np.pi * f[:, None] / C
        e = np.abs(G - np.exp(-1j * k * r) / (4 * np.pi * r)).max()
        back = A.green(f, pts[0], [src])[:, 0]
        pw = A.plane_wave(f, (1, 1, 0), pts)
        e_pw = np.abs(pw - np.exp(-1j * k * (pts @ np.array([1, 1, 0]) / np.sqrt(2)))).max()
        print(f'\n  Green {e:.1e}, reciprocity {np.abs(back - G[:, 0]).max():.1e}, plane wave {e_pw:.1e}')
        self.assertLess(e, 1e-15)
        self.assertLess(np.abs(back - G[:, 0]).max(), 1e-15)
        self.assertLess(e_pw, 1e-14)

    def test_impedance_plane_limits(self):
        # On a rigid plane the image doubles the free field; on a
        # pressure-release plane the pressure is zero; straight above the
        # source the image carries R = (Z - rho c) / (Z + rho c).
        f = np.array([500.])
        src = (0.0, 0.0, 1.0)
        on = np.array([[2.0, 1.0, 0.0]])
        rigid = A.green_impedance_plane(f, src, on, np.inf)
        soft = A.green_impedance_plane(f, src, on, 0)
        Z = 3 * RHO * C
        above = np.array([[0.0, 0.0, 2.0]])
        p = A.green_impedance_plane(f, src, above, Z)
        R = (Z - RHO * C) / (Z + RHO * C)
        expected = A.green(f, src, above) + R * A.green(f, (0, 0, -1.0), above)
        print(f'\n  rigid / free {abs(rigid[0, 0] / A.green(f, src, on)[0, 0]):.6f}, '
              f'soft {abs(soft[0, 0]):.1e}, normal incidence {abs(p - expected).max():.1e}')
        self.assertAlmostEqual(abs(rigid[0, 0] / A.green(f, src, on)[0, 0]), 2, places=12)
        self.assertLess(abs(soft[0, 0]), 1e-15)
        self.assertLess(abs(p - expected).max(), 1e-15)

    def test_sphere_boundary_condition(self):
        # The total field meets dp/dr = j k beta p on the sphere: zero
        # normal derivative for a rigid sphere, and the admittance condition
        # for beta = 0.5 + 0.2j, checked by a one-sided second-order
        # difference in r (points inside the sphere are rejected).
        a, f = 0.1, np.array([300., 1500.])
        th = np.linspace(0.1, np.pi - 0.1, 9)
        u = np.column_stack([np.sin(th), np.zeros_like(th), np.cos(th)])
        k = 2 * np.pi * f[:, None] / C
        h = 1e-6
        for beta in (0, 0.5 + 0.2j):
            tot = [sum(A.plane_wave_on_sphere(f, a, (a + d) * u, beta, max_order=40)) for d in (0, h, 2 * h)]
            dpdr = (-3 * tot[0] + 4 * tot[1] - tot[2]) / (2 * h)
            e = np.abs(dpdr - 1j * k * beta * tot[0]).max() / np.abs(k * tot[0]).max()
            print(f'\n  beta = {beta}: boundary condition met within {e:.1e}', end='')
            self.assertLess(e, 1e-6)

    def test_sphere_rejects_points_inside(self):
        with self.assertRaises(ValueError):
            A.plane_wave_on_sphere([100.], 0.1, [[0, 0, 0.05]])


if __name__ == '__main__':
    unittest.main()
