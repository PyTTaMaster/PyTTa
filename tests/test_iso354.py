"""
Tests for pytta.iso9613 (air absorption, ISO 9613-1) and pytta.iso354
(sound absorption in a reverberation room, ISO 354, with the Sabine and
Eyring reverberation formulas). Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

from pytta import iso354, iso9613

# ISO 9613-1, pure-tone attenuation in dB/km at 20 degrees Celsius, as
# reproduced in Long, Architectural Acoustics, Table 5.1.
FREQS = 1000 * 10 ** (np.array([-12, -9, -6, -3, 0, 3, 6, 9]) / 10)
TABLE = {20: [0.260, 0.712, 1.39, 2.60, 6.53, 21.5, 74.1, 215],
         50: [0.123, 0.445, 1.32, 2.73, 4.66, 9.86, 29.4, 104],
         100: [0.0637, 0.247, 0.895, 2.63, 5.42, 9.21, 19.4, 58.1]}


class TestAirAbsorption(unittest.TestCase):

    def test_iso_9613_1_against_tabulated_values(self):
        for rh, expected in TABLE.items():
            got = 1000 * iso9613.attenuation(FREQS, 20, rh)
            dev = np.max(np.abs(got / expected - 1))
            print(f'\n  20 C {rh} %: largest relative deviation {dev:.1e}')
            # the table has three significant digits
            self.assertLess(dev, 0.005, rh)


class TestISO354(unittest.TestCase):

    def test_speed_of_sound(self):
        self.assertAlmostEqual(iso354.speed_of_sound(20), 343.0)  # Eq. (6)

    def test_absorption_coefficient_round_trip(self):
        # A room of 200 m3 with 30 m2 of absorption; adding 10 m2 of a
        # material with alpha = 0.6 changes T as Sabine says. Equations (5)
        # to (9) must give back alpha, also with air absorption.
        V, A1, S, alpha = 200.0, 30.0, 10.0, 0.6
        for f in (500.0, 4000.0):
            m = iso354.power_attenuation(f, 20, 50)
            c = iso354.speed_of_sound(20)
            T1 = iso354.sabine(V, A1, c, m)
            T2 = iso354.sabine(V, A1 + alpha * S, c, m)
            self.assertAlmostEqual(iso354.absorption_area(T1, V, c, m), A1, 9)
            got = iso354.absorption_coefficient(T1, T2, V, S, 20, 20, m, m)
            self.assertAlmostEqual(got, alpha, 9)

    def test_eyring_against_sabine(self):
        # For the same mean absorption, T_Eyring / T_Sabine is
        # alpha / -ln(1 - alpha): 1 for little absorption, 0.721 at 0.5.
        V, S, c = 200.0, 220.0, 343.0
        for alpha in (1e-6, 0.5):
            ratio = iso354.eyring(V, S, alpha, c) \
                / iso354.sabine(V, S * alpha, c)
            self.assertAlmostEqual(ratio, alpha / -np.log(1 - alpha), 12)
        self.assertAlmostEqual(0.5 / -np.log(0.5), 0.7213, 4)


if __name__ == '__main__':
    unittest.main()


class TestISO354Limits(unittest.TestCase):

    def test_table_1_scales_with_volume(self):
        # Table 1 for 200 m3, times (V / 200)^(2/3).
        np.testing.assert_allclose(iso354.max_room_absorption(200)[[0, 9, 10, 13, 17]],
                                   [6.5, 6.5, 7.0, 9.5, 14.0])
        np.testing.assert_allclose(iso354.max_room_absorption(1600), 4 * iso354.max_room_absorption(200))
        np.testing.assert_allclose(iso354.specimen_area(1600), (10., 48.))
        np.testing.assert_allclose(iso354.specimen_area(150), (10., 12.))

    def test_room_qualification(self):
        # A smooth curve 1 m2 below Table 1 qualifies; a dip of 20 % from
        # the mean of its neighbours does not, nor a band above Table 1.
        A = iso354.max_room_absorption(200) - 1
        self.assertTrue(iso354.room_qualifies(A, 200).all())
        dip = A.copy()
        dip[5] *= 0.8
        self.assertEqual(list(np.flatnonzero(~iso354.room_qualifies(dip, 200))), [5])
        high = A.copy()
        high[17] += 2
        self.assertEqual(list(np.flatnonzero(~iso354.room_qualifies(high, 200))), [17])
