"""
Tests for pytta.iso717.impact, the single-number rating of impact sound
insulation of ISO 717-2. Each test pins one rule.

The text of ISO 717-2 is not at hand; the worked example is the one of
Crocker (ed.), Handbook of Noise and Vibration Control (2007), chapter 106,
Table 13, which follows ISO 717-2.
"""
import os
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

REFERENCE_THIRDS = [62, 62, 62, 62, 62, 62, 61, 60, 59, 58, 57, 54, 51, 48,
                    45, 42]
REFERENCE_OCTAVES = [67, 67, 65, 62, 49]


def unfavourable(Ln, reference, Lnw, at500):
    """Sum of the deviations of Ln above the reference curve shifted to Lnw
    at 500 Hz."""
    shifted = np.array(reference) - reference[at500] + Lnw
    return np.maximum(np.array(Ln) - shifted, 0).sum()


class TestISO717Impact(unittest.TestCase):

    def test_example_of_crocker_table_13(self):
        # Bare floor: Ln,w(CI) = 77 (-11) dB, 29.4 dB of unfavourable
        # deviation. Floor with covering: Ln,w = 40 dB with 29.2 dB; the
        # table prints CI = -3 dB, but its own Ln,sum = 57.9 dB gives
        # 57.9 - 15 - 40 = +2.9 dB, and the floor covering term of the same
        # table (-11 - Cr = -14, Cr = +3) confirms the sign is a misprint.
        bare = [65.5, 66.5, 67, 67.4, 67.8, 68.2, 68.6, 69, 69.3, 69.6, 70,
                70.3, 70.5, 70.7, 71, 71.2]
        covered = [56.2, 50, 47.4, 43.6, 39.1, 36.4, 33.2, 31.7, 31, 29.6,
                   24.7, 23.5, 22.6, 24.3, 19.1, 14.3]
        for Ln, expected, deviation in ((bare, (77, -11), 29.4),
                                        (covered, (40, 3), 29.2)):
            result = pytta.iso717.impact(Ln)
            sum_ = unfavourable(Ln, REFERENCE_THIRDS, result[0], 7)
            print(f'\n{result} vs {expected}; deviations {sum_:.1f} dB vs '
                  f'{deviation} dB')
            self.assertEqual(result, expected)
            self.assertAlmostEqual(sum_, deviation, places=6)

    def test_octave_bands_rate_5_dB_below_the_curve_at_500_Hz(self):
        # Ln 5 dB above the octave reference curve. Shifting it up by 3 dB
        # leaves 2 dB in each of the 5 bands, exactly 10.0 dB, which is
        # allowed; 2 dB would give 15 dB. The curve then reads 65 + 3 dB at
        # 500 Hz and Ln,w is 5 dB less.
        Ln = [r + 5 for r in REFERENCE_OCTAVES]
        Lnw, CI = pytta.iso717.impact(Ln)
        print(f'\nLn,w = {Lnw} dB, deviations '
              f'{unfavourable(Ln, REFERENCE_OCTAVES, Lnw + 5, 2):.1f} dB')
        self.assertEqual(Lnw, 63)

    def test_octave_rating_within_1_dB_of_the_third_octave_rating(self):
        # Crocker, chapter 106: the octave reference curve keeps the ratings
        # from thirds and from octaves within 1 dB. The floors of Table 13,
        # summed to octaves, rate 77 and 41 dB against 77 and 40 dB in
        # thirds. This pins the octave curve, 49 dB at 2 kHz included: with
        # 54 dB there the bare floor would rate 73 dB.
        floors = ([65.5, 66.5, 67, 67.4, 67.8, 68.2, 68.6, 69, 69.3, 69.6, 70,
                   70.3, 70.5, 70.7, 71, 71.2],
                  [56.2, 50, 47.4, 43.6, 39.1, 36.4, 33.2, 31.7, 31, 29.6,
                   24.7, 23.5, 22.6, 24.3, 19.1, 14.3])
        for Ln in floors:
            octaves = [round(10 * np.log10(np.sum(10**(np.array(Ln[i:i + 3])
                                                      / 10))), 1)
                       for i in range(0, 15, 3)]
            thirds, octave = (pytta.iso717.impact(Ln)[0],
                              pytta.iso717.impact(octaves)[0])
            print(f'\nLn,w {thirds} dB in thirds, {octave} dB in octaves')
            self.assertLessEqual(abs(octave - thirds), 1)

    def test_adaptation_terms_sum_up_to_2500_Hz(self):
        # Flat 59.6 dB from 50 Hz to 3 150 Hz (19 thirds). Shifting the curve
        # to 66 dB leaves 28.4 dB of deviation (65 dB would give 33.0 dB).
        # CI sums the 15 bands from 100 Hz to 2 500 Hz:
        # 59.6 + 10 lg 15 - 15 - 66 = -9.64, so -10; with the 3 150 Hz band
        # it would be -9.36, so -9. CI,50-2500 adds 50, 63 and 80 Hz:
        # 59.6 + 10 lg 18 - 15 - 66 = -8.85, so -9.
        result = pytta.iso717.impact([59.6] * 19)
        CI = 59.6 + 10 * np.log10(15) - 15 - 66
        CI50 = 59.6 + 10 * np.log10(18) - 15 - 66
        print(f'\n{result}; CI {CI:.2f}, CI,50-2500 {CI50:.2f}')
        self.assertEqual(result, (66, -10, -9))

    def test_rejects_other_band_counts(self):
        with self.assertRaises(ValueError):
            pytta.iso717.impact([60.0] * 15)


if __name__ == '__main__':
    unittest.main()
