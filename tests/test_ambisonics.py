"""
Tests for pytta.ambisonics, a port of the Higher-Order-Ambisonics library
of Archontis Politis. Each test pins one rule.

The golden data come from golden/make_ambisonics_polarch.m (MATLAB R2026b
with polarch Higher-Order-Ambisonics and Vector-Base-Amplitude-Panning).
"""
import os
import unittest

import numpy as np
import scipy.io as sio

from pytta import ambisonics as A

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


class TestAmbisonics(unittest.TestCase):

    def test_real_sh_are_orthonormal(self):
        # Gauss-Legendre in elevation (12 nodes) and 24 azimuths integrate
        # exactly every product of harmonics up to order 10, so
        # sum w Y Y^T = I. The 21-design does the same with equal weights,
        # (4 pi / N) Y Y^T = I, within the precision of its tabulated points,
        # which falls with the order (4e-10 at order 4, 2.5e-6 at order 10).
        x, w = np.polynomial.legendre.leggauss(12)
        el, az = np.meshgrid(np.degrees(np.arcsin(x)), np.arange(24) * 15., indexing='ij')
        Y = A.real_sh(10, np.column_stack([az.ravel(), el.ravel()]))
        err_gl = np.abs((Y * np.repeat(w, 24) * np.pi / 12) @ Y.T - np.eye(121)).max()
        _, dirs = A.t_design(21)
        Y = A.real_sh(6, dirs)
        err_td = np.abs(4 * np.pi / len(dirs) * Y @ Y.T - np.eye(49)).max()
        print(f'\n  Gauss-Legendre, order 10: {err_gl:.1e}; 21-design, order 6: {err_td:.1e}')
        self.assertLess(err_gl, 1e-13)
        self.assertLess(err_td, 1e-8)

    def test_first_order_is_fuma_bformat(self):
        # FuMa B-format is W = s / sqrt(2), X, Y, Z = s times the unit
        # vector. The first order of an N3D encoding converts to it, scaled
        # by the 1 / sqrt(4 pi) of the orthonormal harmonics, and so does
        # Iceberg's bformat_fuma from a third-order AmbiX, N3D or FuMa scene.
        rng = np.random.default_rng(4)
        s, dirs = rng.uniform(-1, 1, (32, 3)), np.array([[30, 10], [-100, 40], [170, -60]])
        bf = A.encode_bformat(s, dirs) / np.sqrt(4 * np.pi)
        hoa = A.encode(s, dirs, 3)
        errs = [np.abs(A.n3d_to_fuma(hoa[:, :4]) - bf).max(),
                np.abs(A.bformat_fuma(hoa, 'n3d')[0] - bf).max(),
                np.abs(A.bformat_fuma(A.n3d_to_sn3d(hoa), 'ambix')[0] - bf).max(),
                np.abs(A.bformat_fuma(A.n3d_to_fuma(hoa), 'fuma')[0] - bf).max()]
        print('\n  ' + ', '.join(f'{e:.1e}' for e in errs))
        self.assertLess(max(errs), 1e-14)
        self.assertEqual(A.bformat_fuma(hoa, 'n3d')[1], 3)

    def test_rotation_moves_the_encoded_direction(self):
        # Rotating a scene with one source equals encoding the source in
        # the rotated direction, M u, with M = Rx(roll) Ry(-pitch) Rz(-yaw);
        # a positive yaw adds to the azimuth.
        s, u = np.ones((1, 1)), np.array([[25., 15.]])
        yawed = A.rotate(A.encode(s, u, 4), 40, 0, 0)
        e_yaw = np.abs(yawed - A.encode(s, [[65., 15.]], 4)).max()
        M = A._euler_zyx(-np.radians(30), np.radians(20), np.radians(10))
        moved = A._directions((M @ A._unit(u)[0])[None])
        e_any = np.abs(A.rotate(A.encode(s, u, 4), 30, -20, 10)
                       - A.encode(s, moved, 4)).max()
        print(f'\n  yaw only {e_yaw:.1e}, yaw pitch roll {e_any:.1e}')
        self.assertLess(max(e_yaw, e_any), 1e-14)

    def test_matches_polarch(self):
        d = sio.loadmat(os.path.join(GOLDEN, 'ambisonics_polarch.mat'),
                        squeeze_me=True, struct_as_record=False)
        c, D, L, sig3 = d['conv'], d['D'], d['ls_full'], d['sig3']
        hoa2 = np.random.RandomState(2).random_sample((9, 400)).T - 0.5
        self.assertEqual(A.layout_order(d['ls_hemi']), d['order_hemi'])
        self.assertEqual(A.layout_order(L), d['order_full'])
        D_auto, order = A.decoder(L, 'sad')
        self.assertEqual(order, D.auto_order)
        diffs = {
            'real_sh': (A.real_sh(4, d['dirs']), d['Y4']),
            'n3d_to_sn3d': (A.n3d_to_sn3d(sig3), c.n2sn),
            'sn3d_to_n3d': (A.sn3d_to_n3d(sig3), c.sn2n),
            'n3d_to_fuma': (A.n3d_to_fuma(sig3), c.n2fuma),
            'fuma_to_n3d': (A.fuma_to_n3d(sig3), c.fuma2n),
            'acn_to_sid': (A.acn_to_sid(sig3), c.acn2sid),
            'sid_to_acn': (A.sid_to_acn(sig3), c.sid2acn),
            'max_re_weights': (np.concatenate([A.max_re_weights(o) for o in (1, 3, 5)]),
                               np.concatenate(d['maxre'])),
            'sad': (A.decoder(L, 'sad', True, 3)[0], D.sad),
            'mmd': (A.decoder(L, 'mmd', True, 3)[0], D.mmd),
            'epad': (A.decoder(L, 'epad', False, 3)[0], D.epad),
            'allrad': (A.decoder(L, 'allrad', True, 3)[0], D.allrad),
            'csad': (A.decoder(L, 'csad', True, 2)[0], D.csad),
            'sad, order of the layout': (D_auto, D.auto),
            'encode': (A.encode(d['s2'], d['src'], 3), d['enc3']),
            'decode in 3 bands': (A.decode(hoa2, d['Db'], [500, 2000], 48000), d['dec_bands']),
            'encode_bformat': (A.encode_bformat(d['s2'], d['src']), d['bf']),
            'decode_bformat': (A.decode_bformat(d['bf'], d['D1']), d['bf_dec']),
            'rotate_bformat': (A.rotate_bformat(d['bf'], 30, -20, 10), d['rotbf'])}
        worst = {k: np.abs(a - b).max() for k, (a, b) in diffs.items()}
        print('\n  ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
        for k, v in worst.items():
            self.assertLess(v, 1e-13, k)


if __name__ == '__main__':
    unittest.main()
