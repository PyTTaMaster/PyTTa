"""
Tests for the curves drawn by pytta.plot_freq with smoothing and by
pytta.plot_phase. Each test pins one rule on the data of the plotted line.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000


class TestPlotFreq(unittest.TestCase):

    def test_smoothing_is_fractional_octave(self):
        # smooth=n draws the magnitude of smooth_frequency(signal, n), and
        # smooth=True is a third of an octave.
        rng = np.random.default_rng(0)
        sig = pytta.SignalObj(rng.standard_normal(4096), 'time', FS)
        expected = 20 * np.log10(np.abs(
            pytta.smooth_frequency(sig, 3).freqSignal[:, 0]))
        for smooth in (3, True):
            fig = pytta.plot_freq(sig, smooth=smooth, decimalSep='.')
            drawn = fig.axes[0].lines[0].get_ydata()
            self.assertTrue(np.allclose(drawn[1:], expected[1:], atol=1e-4))
            fig.clf()

    def test_phase_of_a_delay(self):
        # A delay of d samples has the phase -360 f d / fs degrees.
        d = 12
        sig = pytta.SignalObj(np.eye(1, 4096, d)[0], 'time', FS)
        fig = pytta.plot_phase(sig, decimalSep='.')
        line = fig.axes[0].lines[0]
        expected = -360 * line.get_xdata() * d / FS
        self.assertTrue(np.allclose(line.get_ydata(), expected, atol=1e-2))
        fig.clf()


if __name__ == '__main__':
    unittest.main()
