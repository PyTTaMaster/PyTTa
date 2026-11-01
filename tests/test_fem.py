"""
Tests for pytta.fem, linear tetrahedra for the Helmholtz equation. Each
test compares with an analytic field: the modes of a rectangular cavity
and plane waves in a tube.
"""
import unittest

import numpy as np

from pytta import fem

C, RHO = 343., 1.21


def box_modes(L, num):
    n = range(5)
    return np.array(sorted(C / 2 * np.sqrt((a / L[0])**2 + (b / L[1])**2 + (d / L[2])**2)
                           for a in n for b in n for d in n)[:num])


class TestFEM(unittest.TestCase):

    def test_cavity_modes_converge_as_h_squared(self):
        # The first ten modes of a 1 x 0.8 x 0.6 m box lie above the exact
        # ones, and the error falls by about four when the elements halve.
        L = (1., 0.8, 0.6)
        exact = box_modes(L, 10)[1:]
        errors = []
        for n in (1, 2):
            nodes, tets, _ = fem.box_mesh(*L, 10 * n, 8 * n, 6 * n)
            f = fem.modes(nodes, tets, 10)[0][1:]
            errors.append(np.abs(f / exact - 1).max())
            self.assertTrue(np.all(f > exact))
        print(f'\n  cavity modes: {errors[0]:.2%} and {errors[1]:.2%}, ratio {errors[0] / errors[1]:.1f}', end='')
        self.assertLess(errors[1], 0.01)
        self.assertTrue(3 < errors[0] / errors[1] < 5)

    def test_tube_with_anechoic_and_rigid_ends(self):
        # A piston moving with u = 1 m/s at x = 0: with rho c at x = 1 m the
        # pressure is rho c u e^(-jkx); with a rigid end it is
        # -j rho c u cos(k (L - x)) / sin(kL).
        nodes, tets, faces = fem.box_mesh(1., 0.05, 0.05, 100, 1, 1)
        k, x = 2 * np.pi * 200 / C, nodes[:, 0]
        cases = (('anechoic', {'x1': 1.}, RHO * C * np.exp(-1j * k * x)),
                 ('rigid', None, -1j * RHO * C * np.cos(k * (1 - x)) / np.sin(k)))
        for name, admittance, exact in cases:
            p = fem.harmonic(nodes, tets, 200., velocity={'x0': 1.}, admittance=admittance,
                             faces=faces, c=C, rho=RHO)
            e = np.abs(p - exact).max() / np.abs(exact).max()
            print(f'\n  tube, {name} end: {e:.1e}', end='')
            self.assertLess(e, 2e-3)


if __name__ == '__main__':
    unittest.main()
