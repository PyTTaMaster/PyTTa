"""
Tests for pytta.rays, stochastic ray tracing in convex rooms. Each test
compares the energy histogram with a result of statistical room acoustics
(Kuttruff, Room Acoustics).
"""
import unittest

import numpy as np

from pytta import rays

C, DT = 343., 1e-3
SRC, RCV = np.array([1.5, 1.2, 1.4]), np.array([4.2, 3.1, 2.2])
V, S = 6 * 5 * 4, 2 * (30 + 24 + 20)


class TestRays(unittest.TestCase):

    def test_direct_sound(self):
        # The direct sound carries E0 / (4 pi r^2) per unit area.
        t, h = rays.trace(rays.shoebox(6, 5, 4, 0.2, 1.), SRC, RCV, 200000, 0.02, seed=1)
        r = np.linalg.norm(RCV - SRC)
        i = int(r / C / DT)
        ratio = h[i - 1:i + 2].sum() * 4 * np.pi * r**2
        print(f'\n  direct sound: {ratio:.3f} of E0 / (4 pi r^2)', end='')
        self.assertLess(abs(ratio - 1), 0.08)

    def test_lossless_diffuse_room_holds_e0_over_v(self):
        # Without absorption the energy stays in the room and spreads
        # evenly: each bin gets the energy density E0 / V times c dt.
        t, h = rays.trace(rays.shoebox(6, 5, 4, 0., 1.), SRC, RCV, 20000, 0.5, seed=2)
        late = h[(t > 0.2) & (t < 0.5)].mean() / (C * DT / V)
        print(f'\n  lossless room: {late:.3f} of E0 c dt / V', end='')
        self.assertLess(abs(late - 1), 0.03)

    def test_decay_lies_between_eyring_and_sabine(self):
        # With alpha = 0.2 on every wall and diffuse reflections the decay
        # falls between Eyring and Sabine, as Kuttruff's correction for
        # the spread of the free paths predicts.
        t, h = rays.trace(rays.shoebox(6, 5, 4, 0.2, 1.), SRC, RCV, 20000, 1.2, seed=3)
        e = np.cumsum(h[::-1])[::-1]
        L = 10 * np.log10(e / e[0])
        i5, i35 = np.argmax(L < -5), np.argmax(L < -35)
        T30 = -60 / np.polyfit(t[i5:i35], L[i5:i35], 1)[0]
        eyring = 0.161 * V / (-S * np.log(0.8))
        sabine = 0.161 * V / (S * 0.2)
        print(f'\n  T30 {T30:.3f} s, Eyring {eyring:.3f} s, Sabine {sabine:.3f} s', end='')
        self.assertTrue(eyring < T30 < sabine)

    def test_convex_room_from_corners(self):
        # The hull of the corners of the shoebox gives its six planes.
        corners = np.array([[x, y, z] for x in (0, 6) for y in (0, 5) for z in (0, 4)])
        room = rays.convex_room(corners, 0.1)
        self.assertEqual(len(room['offsets']), 6)


if __name__ == '__main__':
    unittest.main()
