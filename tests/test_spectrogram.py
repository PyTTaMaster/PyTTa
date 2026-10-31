"""
Tests for the plain spectrogram of pytta.plot_spectrogram
(pytta._plot._calc_spectrogram).

Each test pins one rule: `overlap` is the overlapping fraction of
consecutive windows, each column sits at the centre of its window, and a
sine reads at its peak amplitude in dB whatever the window.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

from pytta import _plot

FS = 8000
W = 1024


def spectrogram(x, overlap=0.5, winType='hann'):
    S, t, f = _plot._calc_spectrogram(x, FS, overlap, winType, W, 1.0)
    return S, t, f


class TestSpectrogram(unittest.TestCase):

    def test_overlap_is_the_overlapping_fraction(self):
        # Hop = W (1 - overlap); every window lies inside the signal, so
        # there are (n - W) // hop + 1 columns, for any overlap in [0, 1).
        x = np.random.default_rng(0).standard_normal(FS)
        for overlap in (0, 0.25, 0.5, 0.75, 0.9):
            hop = W - int(W * overlap)
            S, t, f = spectrogram(x, overlap)
            self.assertEqual(S.shape, (W // 2 + 1, (FS - W) // hop + 1))
            self.assertTrue(np.allclose(np.diff(t), hop / FS))

    def test_columns_sit_at_the_centre_of_their_window(self):
        # The first column is centred at W / 2, and a click is loudest in
        # the column whose centre is nearest to it.
        x = np.zeros(FS)
        x[5000] = 1
        S, t, f = spectrogram(x, 0.75)
        self.assertAlmostEqual(t[0], W / 2 / FS)
        loudest = t[np.argmax(S[W // 8])]
        self.assertLessEqual(abs(loudest - 5000 / FS), 128 / FS)

    def test_sine_reads_at_its_peak_amplitude(self):
        # A sine of amplitude 0.5 centred on a bin reads -6.02 dB re 1, with
        # the coherent gain of each window taken out.
        k = 64
        tt = np.arange(FS) / FS
        x = 0.5 * np.sin(2 * np.pi * k * FS / W * tt)
        for winType in ('hann', 'blackman', 'boxcar'):
            S, t, f = spectrogram(x, 0.5, winType)
            level = np.median(S[k])
            print(f'\n  {winType:8s} {level:+.3f} dB (expected -6.021)')
            self.assertAlmostEqual(level, 20 * np.log10(0.5), delta=0.05)


if __name__ == '__main__':
    unittest.main()
