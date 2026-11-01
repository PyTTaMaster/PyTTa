"""
Tests for pytta.absorbers, layered absorbers by the transfer matrix method
after the Impedance Calculator of the ITA-Toolbox. Each test pins one rule
of the models or of the method.
"""
import unittest

import numpy as np

from pytta import absorbers as A

RHO, C = 1.204, 343.2
F = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000.])


class TestAbsorbers(unittest.TestCase):

    def test_air_layer_on_rigid_backing_does_not_absorb(self):
        # Zs = -j rho c cot(k d), purely reactive, so alpha = 0.
        d = 0.1
        Zs = A.surface_impedance(F, [('air', d)])
        k = 2 * np.pi * F / C
        e = np.abs(Zs - (-1j * RHO * C / np.tan(k * d))).max() / (RHO * C)
        print(f'\n  air layer: Zs against -j rho c cot(kd) {e:.1e}, max alpha '
              f'{np.abs(A.absorption(Zs)).max():.1e}')
        self.assertLess(e, 1e-9)
        self.assertLess(np.abs(A.absorption(Zs)).max(), 1e-12)

    def test_porous_layer_by_the_transfer_matrix(self):
        # One porous layer on a rigid backing: Zs = -j Zc cot(k d) (Aretz,
        # eq. 7.14, with Gamma = j k); a very thick layer tends to Zc.
        Zc, k = A.porous(F, 15000, 'miki')
        Zs = A.surface_impedance(F, [('porous', 'miki', 15000, 0.08)])
        thick = A.surface_impedance(F[3:], [('porous', 'miki', 15000, 5.0)])
        e = np.abs(Zs - (-1j * Zc / np.tan(k * 0.08))).max() / np.abs(Zc).max()
        e_thick = np.abs(thick / Zc[3:] - 1).max()
        print(f'\n  porous layer {e:.1e}; 5 m layer against Zc {e_thick:.1e}')
        self.assertLess(e, 1e-12)
        self.assertLess(e_thick, 1e-6)

    def test_models_tend_to_air_and_miki_constants(self):
        # At f / sigma large Delany-Bazley and Miki tend to air, and Komatsu
        # reaches it at f / sigma = 100, the end of its range; Miki's constants in
        # the model are those of his paper, Eqs. (27) and (28), and its usual
        # SI form (0.0699, 0.107, 0.109, 0.160 with f / sigma) agrees to its
        # rounding.
        f = np.array([1e6])
        for model in ('delany_bazley', 'miki'):
            Zc, k = A.porous(f, 100, model)
            self.assertAlmostEqual(abs(Zc[0] / (RHO * C) - 1), 0, delta=2e-3)
            self.assertAlmostEqual(abs(k[0] / (2 * np.pi * f[0] / C) - 1), 0, delta=2e-3)
        Zc, k = A.porous(np.array([1e4]), 100, 'komatsu')
        self.assertAlmostEqual(abs(Zc[0] / (RHO * C) - 1), 0, places=12)
        self.assertTrue(np.isnan(A.porous(np.array([1e5]), 100, 'komatsu')[0][0]))
        f, sigma = np.array([500.]), 10000.
        Zc, k = A.porous(f, sigma, 'miki')
        X = f / sigma
        Zp = 1 + 0.0699 * X ** -0.632 - 1j * 0.107 * X ** -0.632
        kp = 1 + 0.109 * X ** -0.618 - 1j * 0.160 * X ** -0.618
        e = max(abs(Zc[0] / (RHO * C) - Zp[0]), abs(k[0] / (2 * np.pi * 500 / C) - kp[0]))
        print(f'\n  Miki against its rounded SI form {e:.1e}')
        self.assertLess(e, 3e-3)

    def test_mpp_tends_to_poiseuille(self):
        # Maa: as w -> 0, k_r -> 1 (its second term is linear in x, hence a
        # very low frequency) and k_m -> 4/3 + 0.85 d / t, so the
        # resistance is the Poiseuille resistance 32 eta t / (p d^2) and the
        # mass rho t (4/3 + 0.85 d / t) / p.
        d, t, p, eta = 0.4e-3, 0.5e-3, 0.01, 1.81e-5
        f = np.array([1e-6])
        Z = A.mpp_impedance(f, d, t, p)[0]
        R0 = 32 * eta * t / (p * d ** 2)
        M0 = RHO * t * (4 / 3 + 0.85 * d / t) / p
        print(f'\n  MPP at 1 uHz: R {Z.real:.4f} against {R0:.4f}, '
              f'm {Z.imag / (2 * np.pi * f[0]):.6f} against {M0:.6f}')
        self.assertAlmostEqual(Z.real / R0, 1, delta=1e-5)
        self.assertAlmostEqual(Z.imag / (2 * np.pi * f[0]) / M0, 1, delta=1e-4)

    def test_random_incidence_for_a_real_impedance(self):
        # For a locally reacting surface of real normalised impedance z,
        # Paris' integral is 8/z (1 + 1/(1 + z) - 2/z ln(1 + z)).
        for z in (0.5, 1.0, 3.0, 10.0):
            a = A.random_incidence_absorption(np.array([1000.]), Zs=z * RHO * C, n_angles=2000)[0]
            ref = 8 / z * (1 + 1 / (1 + z) - 2 / z * np.log(1 + z))
            print(f'\n  z = {z}: {a:.6f} against {ref:.6f}', end='')
            self.assertAlmostEqual(a, ref, delta=1e-5)

    def test_rejects_unknown_input(self):
        with self.assertRaises(ValueError):
            A.porous(F, 1e4, 'biot')
        with self.assertRaises(ValueError):
            A.surface_impedance(F, [('foam', 0.1)])


if __name__ == '__main__':
    unittest.main()
