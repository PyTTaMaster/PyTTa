"""
Tests for SignalObj.uncalibrate, which undoes calib_pressure and
calib_voltage: the samples are divided by the correction factor and the
channel goes back to full scale. Each test pins one rule.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000


def calibrator(rms):
    tt = np.arange(FS) / FS
    return pytta.SignalObj(rms * np.sqrt(2) * np.sin(2 * np.pi * 1000 * tt),
                           'time', FS)


class TestUncalibrate(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.sig = pytta.SignalObj(rng.standard_normal((FS, 2)), 'time', FS)
        self.raw = self.sig.timeSignal.copy()
        self.sig.calib_pressure(0, calibrator(0.1), refPrms=1.0)
        self.sig.calib_voltage(1, calibrator(0.2), refVrms=1.0)

    def test_undoes_the_calibration_of_every_channel(self):
        self.sig.uncalibrate()
        self.assertTrue(np.allclose(self.sig.timeSignal, self.raw,
                                    rtol=1e-5, atol=1e-6))
        for ch in self.sig.channels.mapping:
            channel = self.sig.channels[ch]
            self.assertEqual((channel.CF, channel.calibCheck, channel.unit,
                              channel.dBRef), (1, False, 'FS', 1))

    def test_undoes_one_channel(self):
        calibrated = self.sig.timeSignal[:, 1].copy()
        self.sig.uncalibrate(0)
        self.assertTrue(np.allclose(self.sig.timeSignal[:, 0], self.raw[:, 0],
                                    rtol=1e-5, atol=1e-6))
        self.assertTrue(np.array_equal(self.sig.timeSignal[:, 1], calibrated))
        self.assertEqual(self.sig.channels[2].unit, 'V')
        with self.assertRaises(IndexError):
            self.sig.uncalibrate(2)


if __name__ == '__main__':
    unittest.main()
