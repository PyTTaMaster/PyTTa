"""
Tests for the file functions of pytta.functions that follow the ITA-Toolbox
(ita_read_flac, ita_read_caf, ita_read_au, ita_read_blockwise, ita_write_txt).
Each test pins a round trip through a temporary file.
"""
import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

FS = 48000


class TestAudioIO(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        label = 'x'
        t = np.arange(4800) / FS
        self.x = np.column_stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.cos(2 * np.pi * 1000 * t)])
        self.sig = pytta.SignalObj(self.x, 'time', FS)
        self.x32 = self.sig.timeSignal.astype(float)

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_audio_formats_round_trip(self):
        # FLAC at 24 bits keeps the samples within half a step; CAF and AU
        # with float samples keep the float32 samples of SignalObj exactly.
        for name, subtype, tol in (('a.flac', 'PCM_24', 2 ** -23), ('a.caf', 'FLOAT', 0),
                                   ('a.au', 'FLOAT', 0), ('a.wav', 'DOUBLE', 0)):
            pytta.write_audio(self.path(name), self.sig, subtype)
            back = pytta.read_audio(self.path(name))
            e = np.abs(back.timeSignal - self.x32).max()
            print(f'\n  {name} {subtype}: {e:.1e}', end='')
            self.assertEqual(back.samplingRate, FS)
            self.assertLessEqual(e, tol)

    def test_blockwise_reading_rebuilds_the_file(self):
        pytta.write_audio(self.path('b.wav'), self.sig, 'FLOAT')
        blocks = [b.timeSignal for b in pytta.read_blockwise(self.path('b.wav'), 1000)]
        sizes = [b.shape[0] for b in blocks]
        self.assertEqual(sizes, [1000] * 4 + [800])
        np.testing.assert_array_equal(np.vstack(blocks), self.x32)

    def test_text_round_trip(self):
        # Time samples and the real and imaginary parts of the DFT come
        # back; the sampling rate is read from the first column.
        for domain in ('time', 'freq'):
            pytta.write_txt(self.path(f'{domain}.txt'), self.sig, domain)
            back = pytta.read_txt(self.path(f'{domain}.txt'), domain)
            e = np.abs(back.timeSignal - self.x32).max()
            print(f'\n  text, {domain} domain: {e:.1e}, fs {back.samplingRate}', end='')
            self.assertEqual(back.samplingRate, FS)
            self.assertLess(e, 1e-7)
        with open(self.path('freq.txt')) as fh:
            self.assertEqual(len(fh.readline().split()), 1 + 2 * 2)


if __name__ == '__main__':
    unittest.main()
