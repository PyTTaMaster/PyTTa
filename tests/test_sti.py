"""
Tests for pytta.iec60268.sti, the speech transmission index of
IEC 60268-16:2011 by the indirect method (clause 6). Expected values are
worked out here from the equations of the standard; the octave filters may
add a systematic error up to 0.01 STI (A.5.1.2), which is the tolerance.
Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000
TOL = 0.01                                     # A.5.1.2
ALPHA = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
BETA = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
FM = np.array([0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10,
               12.5])


def perfect_channel():
    # no local variables: SignalObj inspects the locals of its caller
    return pytta.SignalObj(np.eye(1, 2 * FS, 100)[0], 'time', FS)


def sti_from_m(m):
    """STI from one modulation transfer ratio per band (A.5.4 to A.5.6)."""
    snr = np.clip(10 * np.log10(m / (1 - m)), -15, 15)
    mti = (snr + 15) / 30
    return np.sum(ALPHA * mti) - np.sum(BETA * np.sqrt(mti[:-1] * mti[1:]))


class TestSTI(unittest.TestCase):

    def test_perfect_channel_gives_one(self):
        # The weights and redundancies of Table A.3 add up to 1.
        for female in (False, True):
            sti, mti, mtf = pytta.iec60268.sti(perfect_channel(),
                                               female=female)
            self.assertGreater(sti, 1 - TOL)
            self.assertLessEqual(sti, 1)

    def test_masking_and_reception_threshold(self):
        # 100 dB in every band: masking of -10 dB from the band below
        # (Table A.1), so m = 1 / 1.1 from 250 Hz up; the threshold of
        # Table A.2 is 54 dB down or more.
        sti, mti, mtf = pytta.iec60268.sti(perfect_channel(),
                                           levels=[100] * 7)
        m = np.r_[1 / (1 + 10 ** ((46 - 100) / 10)), np.full(6, 1 / 1.1)]
        expected = sti_from_m(np.minimum(m, 1 - 1e-12))
        print(f'\n  STI {sti:.4f} (expected {expected:.4f})')
        self.assertAlmostEqual(sti, expected, delta=TOL)

    def test_noise_at_0_db_snr(self):
        # Noise as loud as the signal halves m (6.1), and the masking
        # comes from the total level, 103 dB: m = 0.5 / 1.2 from 250 Hz up.
        sti, mti, mtf = pytta.iec60268.sti(perfect_channel(),
                                           levels=[100] * 7, noise=[100] * 7)
        m = np.r_[0.5, np.full(6, 0.5 / 1.2)]
        expected = sti_from_m(m)
        print(f'\n  STI {sti:.4f} (expected {expected:.4f})')
        self.assertAlmostEqual(sti, expected, delta=TOL)

    def test_mtf_of_an_exponential_decay(self):
        # Diffuse decay with T = 1 s: m(F) = 1 / sqrt(1 + (2 pi F T / 13.8)^2)
        # (Houtgast and Steeneken). A noise decay only follows it on
        # average: its squared envelope fluctuates by about
        # 1 / sqrt(bandwidth x decay time), which in the 125 Hz band makes
        # m wrong by 0.2 or more. The bands from 2 kHz up stay within 0.07
        # (largest of 8 noise seeds).
        rng = np.random.default_rng(0)
        t = np.arange(2 * FS) / FS
        h = rng.standard_normal(t.size) * 10 ** (-3 * t)
        _, _, mtf = pytta.iec60268.sti(pytta.SignalObj(h, 'time', FS))
        theory = 1 / np.sqrt(1 + (2 * np.pi * FM * 1.0 / 13.8) ** 2)
        worst = np.max(np.abs(mtf[:, 4:] - theory[:, None]))
        print(f'\n  largest |m - theory| = {worst:.3f}')
        self.assertLess(worst, 0.07)

    def test_rejects_levels_of_other_lengths(self):
        with self.assertRaises(ValueError):
            pytta.iec60268.sti(perfect_channel(), levels=[60] * 6)
        with self.assertRaises(ValueError):
            pytta.iec60268.sti(perfect_channel(), noise=[40] * 7)


if __name__ == '__main__':
    unittest.main()
