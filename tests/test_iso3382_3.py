"""
Tests for pytta.iso3382_3 against ISO 3382-3:2012: the values of Table 1 and
analytic sound fields along a measurement line.
"""
import unittest

import numpy as np

from pytta import iso3382_3 as iso

R = np.array([2., 3., 4., 6., 8., 11., 16.])


class TestISO338203(unittest.TestCase):

    def test_table_1_and_equation_1(self):
        # The omnidirectional speech of Table 1 sums to 57.4 dB(A), and the
        # level at 1 m is about Lw - 11 dB.
        self.assertAlmostEqual(10 * np.log10(np.sum(10 ** ((iso.SPEECH_LP_1M + iso.A_WEIGHTING) / 10))),
                               57.4, places=1)
        self.assertAlmostEqual(float(iso.source_level_1m(100.)), 100 - 10.99, places=2)

    def test_free_field_decays_6_db_per_doubling(self):
        # A source of 90 dB power in every band in free field: the speech
        # level falls by 20 lg 2 = 6.02 dB per doubling of the distance and
        # is 57.4 - 20 lg 4 = 45.36 dB(A) at 4 m.
        Lp = iso.source_level_1m(np.full(7, 90.)) - 20 * np.log10(R)[:, None]
        LpAS = iso.speech_levels(np.full(7, 90.), Lp)
        D2S, L4 = iso.spatial_decay(R, LpAS)
        L1 = 10 * np.log10(np.sum(10 ** ((iso.SPEECH_LP_1M + iso.A_WEIGHTING) / 10)))
        print(f'\n  free field: D2,S {D2S:.3f} dB, Lp,A,S,4m {L4:.3f} dB', end='')
        self.assertAlmostEqual(D2S, 20 * np.log10(2), places=10)
        self.assertAlmostEqual(L4, L1 - 20 * np.log10(4), places=10)

    def test_positions_outside_2_to_16_m_are_left_out(self):
        # A wall reflection that raises the level at 20 m does not change
        # D2,S, which uses 2 m to 16 m only.
        r = np.r_[R, 20.]
        L = 60 - 5 * np.log2(r)
        L[-1] += 10
        self.assertAlmostEqual(iso.spatial_decay(r, L)[0], 5., places=10)

    def test_distraction_and_privacy_distances(self):
        # STI = 0.8 - 0.05 r: 0.50 at 6 m and 0.20 at 12 m.
        sti = 0.8 - 0.05 * R
        self.assertAlmostEqual(iso.distance_at_sti(R, sti, 0.5), 6., places=10)
        self.assertAlmostEqual(iso.distance_at_sti(R, sti, 0.2), 12., places=10)
        self.assertTrue(np.isnan(iso.distance_at_sti(R, np.full(R.size, 0.6), 0.5)))


if __name__ == '__main__':
    unittest.main()
