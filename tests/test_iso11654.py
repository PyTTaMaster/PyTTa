"""
Tests for pytta.iso11654 with the rules and the examples of ISO 11654:1997.
"""
import unittest

import numpy as np

from pytta import iso11654


class TestISO11654(unittest.TestCase):

    def test_rounding_of_the_practical_coefficients(self):
        # 4.1: x,y2 -> x,y0, x,y3 -> x,y5, x,y7 -> x,y5, x,y8 -> x,y + 0,1,
        # 0,92 -> 0,90, and at most 1,00.
        means = [0.42, 0.43, 0.47, 0.48, 0.92, 1.07]
        third = np.repeat(means, 3)
        np.testing.assert_allclose(iso11654.practical_absorption(third),
                                   [0.40, 0.45, 0.45, 0.50, 0.90, 1.00])

    def test_examples_of_annex_a(self):
        # Figure A.1: alpha_w = 0,60 with an unfavourable deviation at 250
        # Hz; Figure A.2: the same with 1,00 at 500 Hz gives 0,60(M).
        a1 = [0.35, 0.70, 0.65, 0.60, 0.55]
        a2 = [0.35, 1.00, 0.65, 0.60, 0.55]
        self.assertEqual(iso11654.weighted_absorption(a1), (0.60, ''))
        self.assertEqual(iso11654.weighted_absorption(a2), (0.60, 'M'))
        self.assertEqual(iso11654.rating(*iso11654.weighted_absorption(a2)), '0.60(M)')

    def test_full_absorber_and_shape_indicators(self):
        # All ones rate 1.00; a resonator absorbing at 250 Hz only gets L.
        self.assertEqual(iso11654.weighted_absorption([1.0] * 6), (1.00, ''))
        self.assertEqual(iso11654.weighted_absorption([0.9, 0.2, 0.2, 0.2, 0.2]), (0.20, 'L'))


if __name__ == '__main__':
    unittest.main()
