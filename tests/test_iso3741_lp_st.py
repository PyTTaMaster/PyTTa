"""
Tests for pytta.iso3741.Lp_ST, the average sound pressure level over the
microphone positions of ISO 3741:1999, 8.3 and 8.1.5. Each test pins one
rule.
"""
import os
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

FS = 48000


def tone(level, cal=True):
    """1 kHz tone of `level` dB re 20 uPa, flagged as calibrated."""
    label = 'tone'
    t = np.arange(2 * FS) / FS
    amplitude = np.sqrt(2) * 2e-5 * 10**(level / 20)
    sig = pytta.SignalObj(amplitude * np.sin(2 * np.pi * 1000 * t), 'time', FS)
    sig.channels[sig.channels.mapping[0]].calibCheck = cal
    return sig


class TestLpST(unittest.TestCase):

    def test_energy_mean_over_positions_and_arithmetic_deviation(self):
        # Equation (9): Lp = 10 lg(mean 10^(0.1 Lpi)), so 80 and 70 dB give
        # 77.40 dB (the arithmetic mean would be 75 dB). Equation (6): sM is
        # the deviation around the arithmetic mean, 7.07 dB.
        result = pytta.iso3741.Lp_ST([tone(80), tone(70)], 3, 1000, 1000)
        expected = 10 * np.log10((1e8 + 1e7) / 2)
        print(f'\n  Lp = {result.data[0]:.3f} dB (expected {expected:.3f}), '
              f'sM = {result.error[0]:.3f} dB (expected {np.sqrt(50):.3f})')
        self.assertAlmostEqual(result.data[0], expected, delta=0.1)
        self.assertAlmostEqual(result.error[0], np.sqrt(50), delta=0.1)

    def test_rejects_uncalibrated_signal(self):
        with self.assertRaises(ValueError):
            pytta.iso3741.Lp_ST([tone(80), tone(70, cal=False)], 3, 1000, 1000)


if __name__ == '__main__':
    unittest.main()
