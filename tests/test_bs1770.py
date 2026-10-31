"""
Tests for pytta.bs1770: loudness and true peak of ITU-R BS.1770-5, with
the time scales of EBU Tech 3341 and the loudness range of EBU Tech 3342.
The signals are the minimum requirement test signals of those documents,
synthesized here, and each test pins their expected response and
tolerance.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta
from pytta import bs1770

FS = 48000


def tone(segments, channels=2, freq=1000.0, phase=0.0):
    """1 kHz sine of (seconds, dBFS peak level) segments, on all channels."""
    parts = []
    for seconds, level in segments:
        t = np.arange(int(round(seconds * FS))) / FS
        amp = 0.0 if level is None else 10 ** (level / 20)
        parts.append(amp * np.sin(2 * np.pi * freq * t + phase))
    return np.tile(np.concatenate(parts)[:, None], (1, channels))


class TestBS1770(unittest.TestCase):

    def test_k_weighting_at_48_khz_is_that_of_tables_1_and_2(self):
        (b1, a1), (b2, a2) = bs1770.k_weighting(48000)
        self.assertTrue(np.allclose(b1, [1.53512485958697, -2.69169618940638,
                                         1.19839281085285], atol=1e-10))
        self.assertTrue(np.allclose(a1, [1, -1.69065929318241,
                                         0.73248077421585], atol=1e-10))
        self.assertTrue(np.allclose(b2, [1.0, -2.0, 1.0]))
        self.assertTrue(np.allclose(a2, [1, -1.99004745483398,
                                         0.99007225036621], atol=1e-10))

    def test_steady_tones_tech_3341_cases_1_2_and_6(self):
        # M, S, I = -23.0 and -33.0 LUFS for stereo tones; the 5.0 case
        # pins the weights of Table 3 (1.41 on the surrounds).
        for level in (-23.0, -33.0):
            sig = pytta.SignalObj(tone([(20, level)]), 'time', FS)
            I = bs1770.loudness(sig)
            _, M = bs1770.momentary(sig)
            _, S = bs1770.short_term(sig)
            print(f'\n  {level} dBFS: I {I:.2f}, M {np.median(M):.2f}, '
                  f'S {np.median(S):.2f}')
            for value in (I, np.median(M), np.median(S)):
                self.assertAlmostEqual(value, level, delta=0.1)
        levels = [-28.0, -28.0, -24.0, -30.0, -30.0]
        sig = pytta.SignalObj(np.column_stack(
            [tone([(20, L)], 1)[:, 0] for L in levels]), 'time', FS)
        I = bs1770.loudness(sig, weights=[1, 1, 1, 1.41, 1.41])
        self.assertAlmostEqual(I, -23.0, delta=0.1)

    def test_gating_tech_3341_cases_3_4_and_5(self):
        cases = ([(10, -36), (60, -23), (10, -36)],
                 [(10, -72), (10, -36), (60, -23), (10, -36), (10, -72)],
                 [(20, -26), (20.1, -20), (20, -26)])
        for segments in cases:
            sig = pytta.SignalObj(tone(segments), 'time', FS)
            I = bs1770.loudness(sig)
            print(f'\n  {segments}: I {I:.2f}')
            self.assertAlmostEqual(I, -23.0, delta=0.1)

    def test_window_lengths_tech_3341_cases_9_and_12(self):
        # Alternating tones whose energy averages to -23 LUFS over exactly
        # 3 s (S) and 400 ms (M): constant after the first window.
        sig = pytta.SignalObj(tone([(1.34, -20), (1.66, -30)] * 5), 'time', FS)
        t, S = bs1770.short_term(sig)
        self.assertTrue(np.allclose(S[t >= 3.0], -23.0, atol=0.1))
        sig = pytta.SignalObj(tone([(0.18, -20), (0.22, -30)] * 25), 'time', FS)
        t, M = bs1770.momentary(sig)
        self.assertTrue(np.allclose(M[t >= 1.0], -23.0, atol=0.1))

    def test_loudness_range_tech_3342_cases_1_to_4(self):
        cases = (([(20, -20), (20, -30)], 10), ([(20, -20), (20, -15)], 5),
                 ([(20, -40), (20, -20)], 20),
                 ([(20, -50), (20, -35), (20, -20), (20, -35), (20, -50)], 15))
        for segments, expected in cases:
            sig = pytta.SignalObj(tone(segments + [(1.5, None)]), 'time', FS)
            lra = bs1770.loudness_range(sig)
            print(f'\n  LRA {lra:.2f} (expected {expected} +-1)')
            self.assertAlmostEqual(lra, expected, delta=1)

    def test_true_peak_tech_3341_cases_15_to_19(self):
        # Sines between samples: the true peak of an amplitude 0.5 is
        # -6.0 dBTP and of 1.41 is +3.0, within +0.2 / -0.4 dB.
        cases = ((FS / 4, 0.50, 0.0, -6.0), (FS / 4, 0.50, 45.0, -6.0),
                 (FS / 6, 0.50, 60.0, -6.0), (FS / 8, 0.50, 67.5, -6.0),
                 (FS / 4, 1.41, 45.0, 3.0))
        t = np.arange(FS) / FS
        fade = np.minimum(1, np.minimum(t, t[::-1]) / 0.01)
        for freq, amp, phase, expected in cases:
            x = amp * fade * np.sin(2 * np.pi * freq * t + np.radians(phase))
            sig = pytta.SignalObj(np.column_stack((x, x)), 'time', FS)
            tp = bs1770.true_peak(sig)
            print(f'\n  {freq:.0f} Hz {phase} deg: {tp:.2f} dBTP '
                  f'(expected {expected})')
            self.assertGreaterEqual(tp, expected - 0.4)
            self.assertLessEqual(tp, expected + 0.2)


if __name__ == '__main__':
    unittest.main()
