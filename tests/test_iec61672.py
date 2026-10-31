"""
Tests for pytta.iec61672, the sound level meter measures of IEC 61672-1:2013
(frequency weightings A, C and Z, time weightings F and S, Leq, SEL, Lmax,
Lpeak) and the percentile levels of a time-weighted level. Expected values
and acceptance limits are those of the standard. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import scipy.signal as ss

import pytta
from pytta import iec61672

FS = 48000
P94 = np.sqrt(2)                       # sine amplitude of 94 dB SPL, in Pa

# Table 3: nominal frequency, A, C, class 1 acceptance limits (+, -)
TABLE3 = [(10, -70.4, -14.3, 3.0, np.inf), (16, -56.7, -8.5, 2.0, 4.0),
          (31.5, -39.4, -3.0, 1.5, 1.5), (63, -26.2, -0.8, 1.0, 1.0),
          (125, -16.1, -0.2, 1.0, 1.0), (250, -8.6, 0.0, 1.0, 1.0),
          (500, -3.2, 0.0, 1.0, 1.0), (1000, 0.0, 0.0, 0.7, 0.7),
          (2000, 1.2, -0.2, 1.0, 1.0), (4000, 1.0, -0.8, 1.0, 1.0),
          (8000, -1.1, -3.0, 1.5, 2.5), (10000, -2.5, -4.4, 2.0, 3.0),
          (12500, -4.3, -6.2, 2.0, 5.0), (16000, -6.6, -8.5, 2.5, 16.0)]


def in_pascal(signalObj):
    for ch in signalObj.channels.mapping:
        signalObj.channels[ch].unit = 'Pa'
    return signalObj


def sine(seconds, amplitude, freq=1000.0):
    t = np.arange(int(round(seconds * FS))) / FS
    return amplitude * np.sin(2 * np.pi * freq * t)


class TestIEC61672(unittest.TestCase):

    def test_frequency_weightings_within_class_1(self):
        # At the exact frequencies of Table 3 (base-ten, 1000 * 10^(n/10))
        # the response deviates from the design goals by less than the
        # class 1 acceptance limits, at 48 kHz.
        for kind, column in (('A', 1), ('C', 2)):
            sos = iec61672.weighting_filter(kind, FS)
            for row in TABLE3:
                f = 1000 * 10 ** (round(10 * np.log10(row[0] / 1000)) / 10)
                _, h = ss.sosfreqz(sos, worN=np.array([f]), fs=FS)
                dev = 20 * np.log10(abs(h[0])) - row[column]
                self.assertLessEqual(dev, row[3] + 0.05, (kind, row[0]))
                self.assertGreaterEqual(dev, -row[4] - 0.05, (kind, row[0]))

    def test_toneburst_response_of_time_weighting_f(self):
        # Table 4: LAFmax - LA of single 4 kHz tonebursts, and LAE - LA,
        # which is 10 lg(Tb / 1 s).
        table4 = [(1000, 0.0, 0.5, 0.5), (500, -0.1, 0.5, 0.5),
                  (200, -1.0, 0.5, 0.5), (100, -2.6, 1.0, 1.0),
                  (50, -4.8, 1.0, 1.0), (20, -8.3, 1.0, 1.0),
                  (10, -11.1, 1.0, 1.0), (5, -14.1, 1.0, 1.0),
                  (2, -18.0, 1.0, 1.5)]
        steady = pytta.SignalObj(sine(2, P94, 4000), 'time', FS)
        LA = iec61672.leq(in_pascal(steady))[0]
        for ms, ref, up, down in table4:
            burst = np.zeros(2 * FS)
            burst[FS // 2:FS // 2 + int(ms * FS / 1000)] = \
                sine(ms / 1000, P94, 4000)
            sig = in_pascal(pytta.SignalObj(burst, 'time', FS))
            response = iec61672.lmax(sig, time='F')[0] - LA
            self.assertLessEqual(response, ref + up, ms)
            self.assertGreaterEqual(response, ref - down, ms)
            self.assertAlmostEqual(iec61672.sel(sig)[0] - LA,
                                   10 * np.log10(ms / 1000), delta=0.05)

    def test_levels_of_a_steady_94_db_tone(self):
        # 1 Pa rms at 1 kHz: Leq and Lmax 94 dB with A (0 dB at 1 kHz), SEL
        # of 10 s 104 dB, LCpeak of the amplitude sqrt(2) Pa 97.0 dB.
        sig = in_pascal(pytta.SignalObj(sine(10, P94), 'time', FS))
        self.assertAlmostEqual(iec61672.leq(sig)[0], 94.0, delta=0.05)
        self.assertAlmostEqual(iec61672.lmax(sig, time='S')[0], 94.0,
                               delta=0.05)
        self.assertAlmostEqual(iec61672.sel(sig)[0], 104.0, delta=0.05)
        # The peak is that of the steady tone: a 50 ms fade-in keeps the
        # onset transient of the C filter out of it.
        x = sine(2, P94)
        x[:FS // 20] *= np.sin(np.linspace(0, np.pi / 2, FS // 20)) ** 2
        faded = in_pascal(pytta.SignalObj(x, 'time', FS))
        self.assertAlmostEqual(iec61672.lpeak(faded)[0],
                               94 + 20 * np.log10(np.sqrt(2)), delta=0.05)

    def test_percentile_levels(self):
        # 30 s at 60 dB then 30 s at 80 dB: L10 is 80 dB and L90 60 dB.
        x = np.concatenate((sine(30, P94 / 10 ** (34 / 20)),
                            sine(30, P94 / 10 ** (14 / 20))))
        sig = in_pascal(pytta.SignalObj(x, 'time', FS))
        L10, L90 = iec61672.percentile_levels(sig, (10, 90))[:, 0]
        self.assertAlmostEqual(L10, 80.0, delta=0.1)
        self.assertAlmostEqual(L90, 60.0, delta=0.1)

    def test_rejects_unknown_weightings(self):
        sig = pytta.SignalObj(np.zeros(FS), 'time', FS)
        with self.assertRaises(ValueError):
            iec61672.leq(sig, weighting='B')
        with self.assertRaises(ValueError):
            iec61672.lmax(sig, time='I')


if __name__ == '__main__':
    unittest.main()
