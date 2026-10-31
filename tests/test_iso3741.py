"""
Tests for the sound power level of pytta.iso3741, ISO 3741:1999, 8.4. Each
test pins one rule.
"""
import os
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

# A 200 m3 room with 210 m2 of surface, 8 s at 100 Hz and 5 s at 1 000 Hz,
# at 20 degC and 101 325 Pa.
ROOM = dict(T=[8.0, 5.0], V=200.0, S=210.0, f=[100.0, 1000.0],
            temperature=20.0, pressure=101325.0)


class TestISO3741SoundPower(unittest.TestCase):

    def test_direct_method_by_hand(self):
        # Equations (11) and (12), term by term. c = 20.05 sqrt(293)
        # = 343.20 m/s, A = 55.26 V / (c T) = 4.0253 and 6.4405 m2.
        # 100 Hz:   10 lg A = 6.0480, 4.34 A/S = 0.0832,
        #           10 lg(1 + S c / (8 V f)) = 1.6150
        # 1000 Hz:  8.0892, 0.1331, 0.1914
        # both:     -25 lg(427/400 sqrt(273/293) 101325/101300) = -0.3281,
        #           and -6 dB.
        expected = np.array([1.4182, 2.0856])
        LW = pytta.iso3741.sound_power([70.0, 70.0], **ROOM)
        print(f'\nLW - Lp {LW - 70} dB, difference {LW - 70 - expected} dB')
        np.testing.assert_allclose(LW - 70, expected, atol=1e-4)

    def test_pressure_enters_as_25_lg_rho_c(self):
        # B appears only in the impedance term, so halving it raises LW by
        # 25 lg 2 = 7.526 dB in every band.
        LW = pytta.iso3741.sound_power([70.0, 70.0], **ROOM)
        low = pytta.iso3741.sound_power([70.0, 70.0],
                                        **dict(ROOM, pressure=101325.0 / 2))
        print(f'\nLW(B/2) - LW(B) {low - LW} dB, difference '
              f'{low - LW - 25 * np.log10(2)} dB')
        np.testing.assert_allclose(low - LW, 25 * np.log10(2), atol=1e-12)

    def test_comparison_method_agrees_with_the_direct_one(self):
        # Equation (13). A reference source rated in the same room by the
        # direct method gives back the direct result of the source under test,
        # both being normalized to rho c = 400 N s/m3.
        Lp, Lpr = np.array([72.3, 65.1]), np.array([80.0, 78.4])
        LWr = pytta.iso3741.sound_power(Lpr, **ROOM)
        LW = pytta.iso3741.sound_power_comparison(Lp, Lpr, LWr)
        direct = pytta.iso3741.sound_power(Lp, **ROOM)
        print(f'\ncomparison - direct {LW - direct} dB')
        np.testing.assert_allclose(LW, LWr + Lp - Lpr, atol=1e-12)
        np.testing.assert_allclose(LW, direct, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
