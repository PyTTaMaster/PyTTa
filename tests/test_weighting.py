"""
Tests for pytta.weighting against IEC 61672-1:2013, Table 3, whose values
are the expressions of Annex E at the exact frequencies 1000 * 10**(0.1 (n -
30)), rounded to 0.1 dB.
"""
import unittest

import numpy as np

import pytta

A = [-50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9,
     -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5,
     -0.1, -1.1, -2.5, -4.3, -6.6, -9.3]
C = [-6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5, -0.3, -0.2, -0.1] + [0.0] * 9 + \
    [-0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -4.4, -6.2, -8.5, -11.2]


class TestWeighting(unittest.TestCase):

    def test_table_3_in_one_third_octaves(self):
        # 20 Hz to 20 kHz, within the rounding of the table.
        for kind, table in (('A', A), ('C', C)):
            w = pytta.weighting(kind, nth=3)
            self.assertEqual(w.shape, (31, 1))
            e = np.abs(w[:, 0] - table).max()
            print(f'\n  {kind} weighting against Table 3: {e:.3f} dB', end='')
            self.assertLessEqual(e, 0.05 + 1e-9)

    def test_octaves_and_given_frequencies(self):
        # The octave bands are every third one-third octave band from 31.5
        # Hz, and given frequencies are used as they are.
        np.testing.assert_allclose(pytta.weighting('A', nth=1)[:, 0], A[2::3], atol=0.05 + 1e-9)
        np.testing.assert_allclose(pytta.weighting('C', freqs=[1000, 31.5])[:, 0], [0, -3.03])
        with self.assertRaises(ValueError):
            pytta.weighting('Z', nth=3)


if __name__ == '__main__':
    unittest.main()
