"""
Tests for the room tools of pytta.rooms that follow the ITA-Toolbox
(ita_constants, ita_convert_RT_alpha_R_Z, ita_din18041_reverberation_times).
Each test pins one rule.

The golden data come from golden/make_room_tools_ita.m (MATLAB R2026b with
the ITA-Toolbox).
"""
import os
import unittest

import numpy as np
import scipy.io as sio

from pytta import rooms

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


def golden():
    return sio.loadmat(os.path.join(GOLDEN, 'room_tools_ita.mat'), squeeze_me=True)


class TestRoomTools(unittest.TestCase):

    def test_air_constants_match_ita(self):
        # c, rho and z0 of humid air as ita_constants, at three conditions;
        # dry air at 0 C has c = sqrt(1.4 R T) = 331.2 m/s.
        d = golden()
        worst = 0
        for k, (T, phi, p) in enumerate(d['conds']):
            r = rooms.air_constants(T, phi, p)
            worst = max(worst, abs(r['c'] - d['c'][k]), abs(r['rho'] - d['rho'][k]),
                        abs(r['z0'] - d['z0'][k]))
        dry0 = rooms.air_constants(0, 0)['c']
        print(f'\n  worst difference to ITA {worst:.1e}; dry air at 0 C {dry0:.2f} m/s')
        self.assertLess(worst, 1e-9)
        self.assertAlmostEqual(dry0, np.sqrt(1.4 * 8.31 / 0.0289644 * 273.15), places=9)

    def test_conversions_are_consistent(self):
        # T -> alpha -> T returns T by both formulas; alpha = 1 - R^2 and
        # Z = z0 (1 + R) / (1 - R), so R = 0 is alpha = 1 and Z = z0; Eyring
        # tends to Sabine for small alpha.
        V, S, T = 200., 210., np.array([0.5, 1.0, 2.0])
        for formula in ('eyring', 'sabine'):
            a = rooms.convert_room_quantity(T, 'T', 'alpha', V, S, formula)
            back = rooms.convert_room_quantity(a, 'alpha', 'T', V, S, formula)
            np.testing.assert_allclose(back, T, rtol=1e-12)
        R = rooms.convert_room_quantity(0.75, 'alpha', 'R')
        Z = rooms.convert_room_quantity(R, 'R', 'Z')
        print(f'\n  alpha 0.75 -> R {R:.4f} -> Z {abs(Z):.2f} -> alpha '
              f'{rooms.convert_room_quantity(Z, "Z", "alpha"):.6f}')
        self.assertAlmostEqual(float(R), 0.5)
        self.assertAlmostEqual(float(abs(Z)), 3 * 413.3, places=9)
        self.assertAlmostEqual(float(rooms.convert_room_quantity(413.3, 'Z', 'alpha')), 1)
        small_e = rooms.convert_room_quantity(1e-3, 'alpha', 'T', V, S, 'eyring')
        small_s = rooms.convert_room_quantity(1e-3, 'alpha', 'T', V, S, 'sabine')
        self.assertAlmostEqual(small_e / small_s, 1, delta=1e-3)
        with self.assertRaises(ValueError):
            rooms.convert_room_quantity(1., 'T', 'alpha')

    def test_din18041_matches_ita(self):
        # T_soll = 0.37 lg V - 0.14 for speech; the tolerance band of
        # ita_din18041_reverberation_times for speech and music.
        d = golden()
        out = rooms.din18041(1000, 'speech', d['f'])
        music = rooms.din18041(1000, 'music', d['f'])
        e = max(np.abs(out['T_min'] - d['din_speech'][:, 0]).max(),
                np.abs(out['T_max'] - d['din_speech'][:, 1]).max(),
                np.abs(music['T_min'] - d['din_music'][:, 0]).max(),
                np.abs(music['T_max'] - d['din_music'][:, 1]).max())
        print(f"\n  T_soll {out['T_soll']:.4f} s for 1000 m^3; band against ITA {e:.1e}")
        self.assertAlmostEqual(out['T_soll'], 0.37 * 3 - 0.14)
        self.assertLess(e, 1e-12)


if __name__ == '__main__':
    unittest.main()
