"""
Tests for pytta.iso3382.late_lateral_level and late_lateral_average, the
late lateral sound level LJ of ISO 3382-1:2009, A.2.5. The impulse responses
are made of single reflections, so the expected values are exact to the
single precision of SignalObj. Each test pins one rule.
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


class TestLateLateralLevel(unittest.TestCase):

    # Direct sound at 0 (in the null of the figure of eight), lateral
    # reflections at 30 ms and 70 ms (before the 80 ms of A.16), 85 ms and
    # 150 ms. The free-field response at 10 m has amplitude 0.5.
    omni = [(0, 1.0), (0.030, 0.4), (0.070, 0.5), (0.085, 0.3), (0.150, 0.2)]
    eight = [(0.030, 0.4), (0.070, 0.5), (0.085, 0.3), (0.150, 0.2)]
    expected = 10 * np.log10((0.3 ** 2 + 0.2 ** 2) / 0.5 ** 2)  # -2.840 dB

    def level(self, delay=0.0, amplitude=0.5, **kwargs):
        omni = pytta.SignalObj(impulses(*self.omni, delay=delay), 'time', FS)
        eight = pytta.SignalObj(impulses(*self.eight, delay=delay), 'time', FS)
        p10 = pytta.SignalObj(impulses((0.029, amplitude)), 'time', FS)
        return pytta.iso3382.late_lateral_level(omni, eight, p10,
                                                nthOct=None, **kwargs)

    def test_lj_follows_a16_from_80_ms(self):
        LJ = self.level()
        print(f'\nLJ = {LJ:.4f} dB, difference {LJ - self.expected:.2e} dB')
        self.assertAlmostEqual(LJ, self.expected, 6)

    def test_80_ms_count_from_the_direct_sound(self):
        # 10 ms of silence before the direct sound: counted from the first
        # sample, the window would open at 70 ms and take the 70 ms
        # reflection in.
        LJ = self.level(delay=0.010)
        print(f'\nLJ = {LJ:.4f} dB, difference {LJ - self.expected:.2e} dB')
        self.assertAlmostEqual(LJ, self.expected, 6)

    def test_reference_at_another_distance_follows_a4(self):
        # The free-field response at 5 m is twice that at 10 m;
        # LpE,10 = LpE,5 + 20 lg(5/10) gives the same LJ.
        LJ = self.level(amplitude=1.0, distance=5.0)
        print(f'\nLJ = {LJ:.4f} dB, difference {LJ - self.expected:.2e} dB')
        self.assertAlmostEqual(LJ, self.expected, 6)

    def test_one_value_per_octave_band(self):
        # The late lateral part is the free-field response at half the
        # amplitude, so LJ = 20 lg 0.5 in every band.
        omni = pytta.SignalObj(impulses((0, 1.0), (0.2, 0.5)), 'time', FS)
        eight = pytta.SignalObj(impulses((0.2, 0.5)), 'time', FS)
        p10 = pytta.SignalObj(impulses((0.05, 1.0)), 'time', FS)
        LJ = pytta.iso3382.late_lateral_level(omni, eight, p10, nthOct=1,
                                              minFreq=125, maxFreq=4000)
        print(f'\nLJ = {LJ} dB, difference {LJ - 20 * np.log10(0.5)} dB')
        self.assertEqual(len(LJ), 6)
        np.testing.assert_allclose(LJ, 20 * np.log10(0.5), atol=1e-3)


class TestLateLateralAverage(unittest.TestCase):

    def test_average_is_energetic_over_four_octaves(self):
        # A.17: 10 lg(0.25 (3 x 10**-1 + 1)) = -4.881 dB, where the
        # arithmetic mean would be -7.5 dB. Other band counts are refused.
        LJavg = pytta.iso3382.late_lateral_average([-10, -10, -10, 0])
        expected = 10 * np.log10(0.325)
        print(f'\nLJ,avg = {LJavg:.4f} dB, difference '
              f'{LJavg - expected:.2e} dB')
        self.assertAlmostEqual(LJavg, expected, 12)
        with self.assertRaises(ValueError):
            pytta.iso3382.late_lateral_average([-10, -10, -10, 0, -3, -5])


if __name__ == '__main__':
    unittest.main()
