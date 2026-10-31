"""
Test for the channel metadata of SignalObj through .pytta files: the
coordinates and orientation of a channel are saved and restored.
"""
import os
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta


class TestPyttaChannels(unittest.TestCase):

    def test_coordinates_and_orientation_round_trip(self):
        sig = pytta.SignalObj(np.ones(480), 'time', 48000)
        sig.channels[1].coordinates = pytta.classes._base.CoordinateObj(
            point=[1.0, 2.0, 3.0], ref='stage centre', unit='m')
        sig.channels[1].orientation = [0.0, 1.0, 0.0]
        with tempfile.TemporaryDirectory() as folder:
            cwd = os.getcwd()
            os.chdir(folder)            # pytta.save writes next to the file
            try:
                pytta.save('channels.pytta', sig)
                loaded = pytta.load('channels.pytta')
            finally:
                os.chdir(cwd)
        channel = (loaded[0] if isinstance(loaded, list) else loaded).channels[1]
        self.assertEqual(repr(channel.coordinates),
                         repr(sig.channels[1].coordinates))
        self.assertEqual(repr(channel.orientation),
                         repr(sig.channels[1].orientation))


if __name__ == '__main__':
    unittest.main()
