"""
Tests for pytta.generate.stochastic_ir, which follows the statistical model
of Kuttruff, Room Acoustics, 6th ed. (2016), sec. 4.2, as
ita_stochastic_room_impulse_response of the ITA-Toolbox does. Each test pins
one property of the model.
"""
import unittest

import numpy as np

import pytta

FS, V, T, R, C = 48000, 200, 0.8, 3, 343


class TestStochasticIR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = pytta.generate.stochastic_ir(T, V, R, C, FS, seed=4).timeSignal[:, 0].astype(float)

    def test_direct_sound_comes_first(self):
        i = round(R / C * FS)
        self.assertFalse(self.x[:i].any())
        self.assertAlmostEqual(self.x[i], 1 / R, places=6)

    def test_reflection_density(self):
        # Number of reflections between 20 and 40 ms against the integral of
        # 4 pi c^3 t^2 / V (eq. 4.6), where they rarely share a sample.
        t1, t2 = 0.02, 0.04
        count = np.count_nonzero(self.x[round(t1 * FS):round(t2 * FS)])
        expected = 4 * np.pi * C**3 * (t2**3 - t1**3) / (3 * V)
        print(f'\n  reflections: {count}, expected {expected:.0f}', end='')
        self.assertLess(abs(count - expected), 4 * np.sqrt(expected))

    def test_decay_gives_the_reverberation_time(self):
        # T30 from the backward integrated energy (-5 to -35 dB).
        e = np.cumsum(self.x[::-1] ** 2)[::-1]
        L = 10 * np.log10(e / e[0])
        i5, i35 = np.argmax(L < -5), np.argmax(L < -35)
        slope = np.polyfit(np.arange(i5, i35) / FS, L[i5:i35], 1)[0]
        print(f'\n  T30 {-60 / slope:.3f} s for {T} s', end='')
        self.assertLess(abs(-60 / slope / T - 1), 0.02)


if __name__ == '__main__':
    unittest.main()
