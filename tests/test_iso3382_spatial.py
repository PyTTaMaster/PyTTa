"""
Tests for pytta.iso3382.lateral_fraction and pytta.iso3382.iacc, the early
lateral energy fractions (ISO 3382-1:2009, A.2.4) and the interaural cross
correlation coefficient (Annex B). The impulse responses are made of single
reflections, so the expected values are exact. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000
N = FS // 2


def impulses(*arrivals, delay=0.0):
    """Impulse response with (time in s, amplitude) arrivals."""
    h = np.zeros(N)
    for t, a in arrivals:
        h[int(round((t + delay) * FS))] += a
    return h


class TestLateralFraction(unittest.TestCase):

    # Direct sound at 0 (in the null of the figure of eight), lateral
    # reflections at 3 ms (before the 5 ms of A.14), 20 ms and 40 ms (the
    # last 60 degrees off the lateral axis, so cos = 0.5), and one at 100 ms,
    # after the 80 ms window.
    omni = [(0, 1.0), (0.003, 0.3), (0.020, 0.5), (0.040, 0.4), (0.100, 0.6)]
    eight = [(0.003, 0.3), (0.020, 0.5), (0.040, 0.2), (0.100, 0.6)]
    energy = 1 + 0.3 ** 2 + 0.5 ** 2 + 0.4 ** 2

    def fractions(self, delay=0.0):
        omni = pytta.SignalObj(impulses(*self.omni, delay=delay), 'time', FS)
        eight = pytta.SignalObj(impulses(*self.eight, delay=delay), 'time', FS)
        return pytta.iso3382.lateral_fraction(omni, eight, nthOct=None)

    def test_jlf_and_jlfc_follow_a14_and_a15(self):
        JLF, JLFC = self.fractions()
        self.assertAlmostEqual(JLF, (0.5 ** 2 + 0.2 ** 2) / self.energy, 6)
        self.assertAlmostEqual(JLFC, (0.5 * 0.5 + 0.2 * 0.4) / self.energy, 6)

    def test_time_starts_at_the_direct_sound(self):
        # Silence before the direct sound does not change the fractions.
        self.assertTrue(np.allclose(self.fractions(delay=0.010),
                                    self.fractions(), atol=1e-7))


class TestIACC(unittest.TestCase):

    def iacc(self, left, right, **kwargs):
        binaural = pytta.SignalObj(np.column_stack((left, right)), 'time', FS)
        return pytta.iso3382.iacc(binaural, nthOct=None, **kwargs)

    def test_lag_range_is_below_1_ms(self):
        # The same arrivals 0.5 ms apart correlate fully; 2 ms apart, not
        # at all. A sign change does not matter (B.2 takes |IACF|).
        left = impulses((0, 1.0), (0.012, 0.7), (0.031, -0.4))
        self.assertAlmostEqual(self.iacc(left, left), 1, 6)
        self.assertAlmostEqual(self.iacc(left, -left), 1, 6)
        self.assertAlmostEqual(self.iacc(left, np.roll(left, 24)), 1, 6)
        self.assertAlmostEqual(self.iacc(left, np.roll(left, 96)), 0, 6)

    def test_time_window_of_early_and_late_parts(self):
        # Identical early parts and late parts 3 ms apart: IACC is 1 for
        # 0 to 80 ms and 0 for 80 ms to the end.
        early = [(0, 1.0), (0.015, 0.5)]
        late = [(0.150, 0.3)]
        left = impulses(*early, *late)
        right = impulses(*early, *[(t + 0.003, a) for t, a in late])
        self.assertAlmostEqual(self.iacc(left, right, t1=0, t2=0.08), 1, 6)
        self.assertAlmostEqual(self.iacc(left, right, t1=0.08), 0, 6)

    def test_one_value_per_octave_band(self):
        rng = np.random.default_rng(0)
        t = np.arange(N) / FS
        h = rng.standard_normal(N) * np.exp(-t / 0.1)
        binaural = pytta.SignalObj(np.column_stack((h, h)), 'time', FS)
        values = pytta.iso3382.iacc(binaural, nthOct=1, minFreq=125,
                                    maxFreq=4000)
        self.assertEqual(len(values), 6)
        self.assertTrue(np.allclose(values, 1, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
