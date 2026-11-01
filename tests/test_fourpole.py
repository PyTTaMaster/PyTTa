"""
Tests for pytta.fourpole, which follows itaFourpole and ita_make_fourpole of
the ITA-Toolbox. Each test pins one rule against an analytic result.
"""
import unittest

import numpy as np

from pytta import fourpole

F = np.linspace(10, 1000, 200)


class TestFourpole(unittest.TestCase):

    def test_mass_on_a_spring(self):
        # A mass on a grounded spring has Z = j omega m + 1 / (j omega n),
        # and a free output leaves only the mass.
        m, n = 0.1, 1e-4
        A = fourpole.chain(fourpole.element('mass', m, F), fourpole.element('spring', n, F))
        jw = 2j * np.pi * F
        e = np.abs(fourpole.input_impedance(A) / (jw * m + 1 / (jw * n)) - 1).max()
        e_free = np.abs(fourpole.input_impedance(A, 0) / (jw * m) - 1).max()
        print(f'\n  mass on spring: blocked {e:.1e}, free {e_free:.1e}', end='')
        self.assertLess(max(e, e_free), 1e-12)

    def test_impedance_matrix_is_reciprocal(self):
        # A chain of passive elements has det A = 1, so Z12 = Z21, and the
        # conversion goes back.
        A = fourpole.chain(fourpole.element('mass', 0.2, F), fourpole.element('damper', 50, F),
                           fourpole.element('spring', 3e-4, F), fourpole.element('shunt', 20 + 5j, F))
        Z = fourpole.a_to_z(A)
        e = np.abs(Z[:, 0, 1] - Z[:, 1, 0]).max() / np.abs(Z[:, 1, 0]).max()
        back = np.abs(fourpole.z_to_a(Z) - A).max() / np.abs(A).max()
        print(f'\n  reciprocity {e:.1e}, A -> Z -> A {back:.1e}', end='')
        self.assertLess(max(e, back), 1e-12)

    def test_gyrator_twice_and_transformer(self):
        G = fourpole.element('gyrator', 7.5, F)
        T = fourpole.element('transformer', 3, F)
        np.testing.assert_allclose(fourpole.chain(G, G), np.broadcast_to(np.eye(2), G.shape), atol=1e-15)
        # A load Z behind a transformer of ratio t is seen as t^2 Z.
        np.testing.assert_allclose(fourpole.input_impedance(T, 4 + 1j), 9 * (4 + 1j))


if __name__ == '__main__':
    unittest.main()
