"""
Test for SignalObj.calib_pressure: the correction factor is the reference
rms over the rms of the calibrator recording. On NumPy 2 that ratio is a
NumPy float32, which the channel must accept.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta

FS = 48000


class TestCalibration(unittest.TestCase):

    def test_correction_factor_from_a_calibrator(self):
        tt = np.arange(FS) / FS
        calibrator = pytta.SignalObj(
            0.1 * np.sqrt(2) * np.sin(2 * np.pi * 1000 * tt), 'time', FS)
        sig = pytta.SignalObj(np.ones(FS), 'time', FS)
        sig.calib_pressure(0, calibrator, refPrms=1.0)
        self.assertAlmostEqual(sig.channels[1].CF, 10, places=5)
        self.assertTrue(np.allclose(sig.timeSignal, 10, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
