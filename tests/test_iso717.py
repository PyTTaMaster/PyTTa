"""
Tests for pytta.iso717.airborne, the single-number rating of airborne
sound insulation of ISO 717-1:2013. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

# ISO 717-1:2013, Annex C, Table C.1: sound reduction index of a building
# element, one-third-octave bands from 100 Hz to 3 150 Hz.
R_ANNEX_C = [20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6, 28.0, 30.5, 31.8,
             32.5, 33.4, 33.0, 31.0, 25.5]


class TestISO717(unittest.TestCase):

    def test_example_of_annex_c(self):
        # Rw(C;Ctr) = 30 (-2; -3) dB, with 31.8 dB of unfavourable deviation
        self.assertEqual(pytta.iso717.airborne(R_ANNEX_C), (30, -2, -3))

    def test_deviation_limit_is_inclusive(self):
        # Octave bands: R 5 dB above the reference curve. Shifting it up by
        # 7 dB gives 2 dB below R in each of the 5 bands, exactly 10.0 dB,
        # which is allowed; 8 dB would give 15 dB.
        reference = [36, 45, 52, 55, 56]
        Rw, C, Ctr = pytta.iso717.airborne([r + 5 for r in reference])
        self.assertEqual(Rw, 52 + 7)

    def test_input_is_rounded_half_up_to_one_decimal(self):
        # The standard rounds R to 0.1 dB before rating, halves up. These
        # five bands end in 5, and round() of Python, which rounds the
        # binary value, takes each of them 0.1 dB down: the deviations would
        # sum 32.3 dB and Rw would drop to 29. Rounded half up, the curve is
        # that of Annex C.
        R = list(R_ANNEX_C)
        for band, value in ((5, 22.65), (8, 27.95), (9, 30.45), (14, 30.95),
                            (15, 25.45)):
            R[band] = value
        self.assertEqual(pytta.iso717.airborne(R), (30, -2, -3))

    def test_rejects_other_band_counts(self):
        with self.assertRaises(ValueError):
            pytta.iso717.airborne(R_ANNEX_C[:-1])


if __name__ == '__main__':
    unittest.main()
