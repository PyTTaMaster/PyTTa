"""
Tests for the fractional octave bands of pytta.utils.fractional_octave_frequencies
and the filters of OctFilter against IEC 61260-1:2014: band edges from the
exact mid-band frequencies (Eq. 2, 4 and 5), nominal frequencies as labels
(Annex E), and the class 1 limits on relative attenuation (Table 1 and
Table F.1).
"""
import unittest

import numpy as np
import scipy.signal as ss

import pytta
from pytta.utils.freq import fractional_octave_frequencies

FS = 48000
G = 10 ** 0.3


def class1_violations(nthOct, freqRange, points):
    # Breakpoints (normalized frequency, minimum, maximum) of the relative
    # attenuation, evaluated around the exact mid-band frequency of each band.
    of = pytta.OctFilter(order=4, nthOct=nthOct, samplingRate=FS, minFreq=freqRange[0],
                         maxFreq=freqRange[1], refFreq=1000, base=10)
    bands = fractional_octave_frequencies(nthOct, freqRange)
    bad = []
    for b, (lower, nominal, upper) in enumerate(bands):
        x = np.round(np.log(nominal / 1000) / np.log(G) * nthOct)
        fm = 1000 * G ** (x / nthOct)            # exact, from the band number
        sos = np.ascontiguousarray(of.sos[:, :, b])
        f = fm * np.array([1.0] + [w for w, _, _ in points])
        keep = f < FS / 2
        att = -20 * np.log10(np.abs(ss.sosfreqz(sos, worN=f[keep], fs=FS)[1]))
        rel = att[1:] - att[0]
        for (w, lo, hi), a in zip(np.array(points, dtype=object)[keep[1:]], rel):
            if not lo <= a <= hi:
                bad.append((nominal, w, round(a, 2)))
    return bad


class TestBandEdges(unittest.TestCase):

    def test_edges_come_from_the_exact_mid_band_frequencies(self):
        # Octave and one-third octave: fm = 1000 G^(x/b), f1,2 = fm G^(-+1/(2b)).
        for b, rng in ((1, (31.5, 16000)), (3, (25, 20000))):
            F = fractional_octave_frequencies(b, rng)
            x = np.round(np.log(np.sqrt(F[:, 0] * F[:, 2]) / 1000) / np.log(G) * b)
            fm = 1000 * G ** (x / b)
            np.testing.assert_allclose(F[:, 0], fm * G ** (-1 / (2 * b)), rtol=1e-12)
            np.testing.assert_allclose(F[:, 2], fm * G ** (1 / (2 * b)), rtol=1e-12)
            self.assertTrue(np.all(np.diff(x) == 1))
        # The nominal frequencies stay as labels, 125 Hz for 125.89 Hz.
        self.assertEqual(fractional_octave_frequencies(1, (100, 200))[0, 1], 125.)
        self.assertEqual(fractional_octave_frequencies(3, (0.6, 0.7))[0, 1], 0.63)

    def test_octave_filters_meet_class_1(self):
        e = 1e-4
        points = [(G ** -4, 70, np.inf), (G ** -3, 60, np.inf), (G ** -2, 40.5, np.inf),
                  (G ** -1, 16.6, np.inf), (G ** -0.5 * (1 - e), 1.2, np.inf),
                  (G ** -0.5 * (1 + e), -0.4, 5.3), (G ** (-3 / 8), -0.4, 1.4),
                  (G ** (-1 / 4), -0.4, 0.7), (G ** (-1 / 8), -0.4, 0.5),
                  (G ** (1 / 8), -0.4, 0.5), (G ** (1 / 4), -0.4, 0.7), (G ** (3 / 8), -0.4, 1.4),
                  (G ** 0.5 * (1 - e), -0.4, 5.3), (G ** 0.5 * (1 + e), 1.2, np.inf),
                  (G, 16.6, np.inf), (G ** 2, 40.5, np.inf), (G ** 3, 60, np.inf), (G ** 4, 70, np.inf)]
        self.assertEqual(class1_violations(1, (31.5, 16000), points), [])

    def test_third_octave_filters_meet_class_1(self):
        points = [(0.18546, 70, np.inf), (0.32748, 60, np.inf), (0.53143, 40.5, np.inf),
                  (0.77257, 16.6, np.inf), (0.89125 - 1e-4, 1.2, np.inf), (0.89125 + 1e-4, -0.4, 5.3),
                  (0.91958, -0.4, 1.4), (0.94719, -0.4, 0.7), (0.97402, -0.4, 0.5),
                  (1.02667, -0.4, 0.5), (1.05575, -0.4, 0.7), (1.08746, -0.4, 1.4),
                  (1.12202 - 1e-4, -0.4, 5.3), (1.12202 + 1e-4, 1.2, np.inf), (1.29437, 16.6, np.inf),
                  (1.88173, 40.5, np.inf), (3.05365, 60, np.inf), (5.39195, 70, np.inf)]
        self.assertEqual(class1_violations(3, (25, 20000), points), [])


if __name__ == '__main__':
    unittest.main()
