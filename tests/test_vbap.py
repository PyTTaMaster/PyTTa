"""
Tests for pytta.vbap, vector base amplitude panning (Pulkki 1997), a port of
the VBAP library of Archontis Politis. Each test pins one rule.

The golden data come from golden/make_vbap_polarch.m (MATLAB R2026b with
polarch Vector-Base-Amplitude-Panning, the copy that Iceberg installs);
their loudspeaker indices are 1-based.
"""
import os
import unittest

import numpy as np
import scipy.io as sio

from pytta import vbap

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


def unit(azi, elev=0.):
    a, e = np.radians(azi), np.radians(elev)
    return np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])


class TestVBAP(unittest.TestCase):

    def test_stereo_pair_follows_the_tangent_law(self):
        # Loudspeakers at +-30 deg: (g1 - g2) / (g1 + g2) = tan(theta) /
        # tan(30 deg), with g1^2 + g2^2 = 1 (Pulkki 1997, eq. 4 and 6).
        ls = [30, -30]
        groups = vbap.loudspeaker_pairs(ls)
        inv = vbap.inverse_matrices(ls, groups)
        theta = np.linspace(-29, 29, 59)
        g = vbap.gains(theta, groups, inv)
        law = (g[:, 0] - g[:, 1]) / (g[:, 0] + g[:, 1])
        err = np.max(np.abs(law - np.tan(np.radians(theta)) / np.tan(np.radians(30))))
        print(f'\n  max |tangent law error| = {err:.1e}, '
              f'max |norm - 1| = {np.max(np.abs(np.linalg.norm(g, axis=1) - 1)):.1e}')
        self.assertLess(err, 1e-12)
        np.testing.assert_allclose(np.linalg.norm(g, axis=1), 1, atol=1e-12)

    def test_3d_gains_point_at_the_source(self):
        # VBAP definition: the gains of the active triplet weight the
        # loudspeaker vectors into a vector along the source direction, all
        # gains are non-negative and their norm is 1. A source on a
        # loudspeaker gets that loudspeaker alone.
        ls = np.array([[0, 0], [90, 0], [180, 0], [-90, 0], [45, 45],
                       [135, 45], [-135, 45], [-45, 45], [0, 90]])
        groups = vbap.loudspeaker_triplets(ls)
        inv = vbap.inverse_matrices(ls, groups)
        rng = np.random.default_rng(1)
        src = np.column_stack([rng.uniform(-180, 180, 200), rng.uniform(1, 89, 200)])
        g = vbap.gains(src, groups, inv)
        L = np.array([unit(*d) for d in ls])
        p = g @ L
        u = np.array([unit(*s) for s in src])
        cross = np.max(np.linalg.norm(np.cross(p, u), axis=1))
        print(f'\n  max |p x u| = {cross:.1e} over 200 sources')
        self.assertLess(cross, 1e-12)
        self.assertTrue(np.all(g >= 0))
        np.testing.assert_allclose(np.linalg.norm(g, axis=1), 1, atol=1e-12)
        on = vbap.gains(ls[[4]], groups, inv)[0]
        self.assertAlmostEqual(on[4], 1, places=12)
        self.assertAlmostEqual(np.sum(np.delete(on, 4)**2), 0, places=12)

    def test_matches_polarch(self):
        # Same layouts and sources: pairs and triplets (1-based there), the
        # inverse matrices (stacked by columns there) and the gains, with and
        # without a spread of 20 deg (MDAP).
        d = sio.loadmat(os.path.join(GOLDEN, 'vbap_polarch.mat'), squeeze_me=True)
        pairs = vbap.loudspeaker_pairs(d['ls2'])
        tri = vbap.loudspeaker_triplets(d['ls3'])
        np.testing.assert_array_equal(pairs + 1, d['pairs'])
        np.testing.assert_array_equal(tri + 1, d['tri'])
        inv2 = vbap.inverse_matrices(d['ls2'], pairs)
        inv3 = vbap.inverse_matrices(d['ls3'], tri)
        diffs = {
            'inv2': np.abs(inv2.transpose(0, 2, 1).reshape(len(pairs), -1) - d['inv2']).max(),
            'inv3': np.abs(inv3.transpose(0, 2, 1).reshape(len(tri), -1) - d['inv3']).max(),
            'g2': np.abs(vbap.gains(d['src2'], pairs, inv2) - d['g2']).max(),
            'g2s': np.abs(vbap.gains(d['src2'], pairs, inv2, 20) - d['g2s']).max(),
            'g3': np.abs(vbap.gains(d['src3'], tri, inv3) - d['g3']).max(),
            'g3s': np.abs(vbap.gains(d['src3'], tri, inv3, 20) - d['g3s']).max()}
        print('\n  ' + ', '.join(f'{k} {v:.1e}' for k, v in diffs.items()))
        for v in diffs.values():
            self.assertLess(v, 1e-12)


if __name__ == '__main__':
    unittest.main()
