"""
Tests for the editing utilities of pytta.functions that follow the ITA-Toolbox
(ita_envelope, ita_start_IR, ita_time_shift, ita_append, ita_normalize_dat,
ita_normalize_spk, ita_amplify_to, ita_zerophase, ita_randomize_phase,
ita_quantize). Each test pins one rule.

The golden data come from golden/make_dsp_edit_ita.m (MATLAB R2026b with the
ITA-Toolbox); its indices are 1-based.

SignalObj inspects the caller's frame (classes/_instanceinfo.py) and fails
when the caller's first local variable is an array, so each test starts
with a scalar local.
"""
import os
import unittest

import numpy as np
import scipy.io as sio

os.environ.setdefault('MPLBACKEND', 'Agg')

import pytta

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


def golden():
    return sio.loadmat(os.path.join(GOLDEN, 'dsp_edit_ita.mat'), squeeze_me=True)


class TestEdit(unittest.TestCase):

    def test_start_ir_matches_ita(self):
        # ISO 3382-1, A.3: the last sample 20 dB below the peak before it,
        # here after a weak lead-in; ITA indices are 1-based.
        label = 'start'
        d = golden()
        ir = pytta.SignalObj(d['ir'].astype(float), 'time', int(d['fs']))
        start = pytta.start_ir(ir)
        print(f'\n  start {start}, ITA {d["start"] - 1}')
        np.testing.assert_array_equal(start, d['start'] - 1)

    def test_envelope_of_an_am_tone(self):
        # The envelope of (1 + m cos(2 pi fm t)) cos(2 pi fc t) is
        # 1 + m cos(2 pi fm t) away from the ends, and the ITA envelope of
        # the same samples agrees.
        label = 'envelope'
        d = golden()
        fs = int(d['fs'])
        t = np.arange(2048) / fs
        am = (1 + 0.5 * np.cos(2 * np.pi * 50 * t)) * np.cos(2 * np.pi * 2000 * t)
        env = pytta.envelope(pytta.SignalObj(am, 'time', fs)).timeSignal[:, 0]
        mid = slice(200, -200)
        e_rule = np.abs(env[mid] - (1 + 0.5 * np.cos(2 * np.pi * 50 * t[mid]))).max()
        e_ita = np.abs(env - d['env']).max()
        print(f'\n  max |env - (1 + m cos)| {e_rule:.1e} (away from the ends), against ITA {e_ita:.1e}')
        self.assertLess(e_rule, 1e-2)
        self.assertLess(e_ita, 1e-6)

    def test_time_shift(self):
        # An integer shift is a cyclic roll; half a sample twice is one
        # sample; None brings the start of the response to sample 0.
        label = 'shift'
        fs = 48000
        x = np.zeros(256)
        x[40:60] = np.hanning(20)
        sig = pytta.SignalObj(x, 'time', fs)
        rolled = pytta.time_shift(sig, 5, unit='samples').timeSignal[:, 0]
        half = pytta.time_shift(pytta.time_shift(sig, 0.5, 'samples'), 0.5, 'samples').timeSignal[:, 0]
        auto = pytta.time_shift(sig)
        e_half = np.abs(half - np.roll(x, 1)).max()
        print(f'\n  roll {np.abs(rolled - np.roll(x, 5)).max():.1e}, two half samples {e_half:.1e}, '
              f'start after auto {pytta.start_ir(auto)[0]}')
        np.testing.assert_allclose(rolled, np.roll(x, 5), atol=1e-7)
        self.assertLess(e_half, 1e-6)
        self.assertEqual(pytta.start_ir(auto)[0], 0)

    def test_levels(self):
        # normalize brings the peak (time or spectrum) to 1 and returns it;
        # amplify_to sets the mean RMS level re 20 uPa for channels in Pa.
        label = 'levels'
        fs = 48000
        t = np.arange(fs) / fs
        x = np.column_stack([0.2 * np.sin(2 * np.pi * 1000 * t), 0.05 * np.sin(2 * np.pi * 500 * t)])
        sig = pytta.SignalObj(x, 'time', fs)
        for num in sig.channels.mapping:
            sig.channels[num].unit = 'Pa'
        normed, gain = pytta.normalize(sig)
        per, gains = pytta.normalize(sig, per_channel=True)
        spec, _ = pytta.normalize(sig, domain='freq')
        at70 = pytta.amplify_to(sig, 70).timeSignal
        level = 20 * np.log10(np.mean(np.sqrt(np.mean(at70 ** 2, axis=0))) / 2e-5)
        print(f'\n  peaks {np.abs(normed.timeSignal).max():.6f}, {np.abs(per.timeSignal).max(axis=0)}, '
              f'gain {gain[0]:.4f}; level {level:.6f} dB')
        self.assertAlmostEqual(np.abs(normed.timeSignal).max(), 1, places=6)
        np.testing.assert_allclose(np.abs(per.timeSignal).max(axis=0), 1, atol=1e-6)
        np.testing.assert_allclose(gains, [0.2, 0.05], rtol=1e-6)
        self.assertAlmostEqual(np.abs(np.fft.rfft(spec.timeSignal, axis=0)).max(), 1, places=5)
        self.assertAlmostEqual(level, 70, delta=1e-3)   # SignalObj stores float32

    def test_phase_edits_keep_the_magnitude(self):
        # zero_phase and randomize_phase keep the magnitude spectrum; the
        # zero-phase signal is even (x[n] = x[N - n]).
        label = 'phase'
        fs = 48000
        x = np.random.default_rng(3).standard_normal(1024)
        sig = pytta.SignalObj(x, 'time', fs)
        mag = np.abs(np.fft.rfft(x))
        z = pytta.zero_phase(sig).timeSignal[:, 0]
        r = pytta.randomize_phase(sig, seed=1).timeSignal[:, 0]
        e_z = np.abs(np.abs(np.fft.rfft(z)) - mag).max()
        e_r = np.abs(np.abs(np.fft.rfft(r)) - mag).max()
        print(f'\n  magnitude kept: zero phase {e_z:.1e}, random phase {e_r:.1e}')
        self.assertLess(max(e_z, e_r) / mag.max(), 1e-5)
        np.testing.assert_allclose(z[1:], z[1:][::-1], atol=1e-6)

    def test_append_and_quantize(self):
        # append concatenates in time; quantize rounds to 2^(b-1) + 1 steps
        # across the peak, as ITA does with the same samples.
        label = 'append'
        d = golden()
        fs = int(d['fs'])
        a = pytta.SignalObj(np.ones(100), 'time', fs)
        b = pytta.SignalObj(np.zeros(50), 'time', fs)
        self.assertEqual(pytta.append(a, b).numSamples, 150)
        q = pytta.quantize(pytta.SignalObj(d['ir'][:, 0].astype(float), 'time', fs), bits=8)
        e = np.abs(q.timeSignal[:, 0] - d['q']).max()
        print(f'\n  quantize against ITA {e:.1e}')
        self.assertLess(e, 1e-7)
        with self.assertRaises(ValueError):
            pytta.append(a, pytta.SignalObj(np.zeros(50), 'time', 48000))


if __name__ == '__main__':
    unittest.main()
