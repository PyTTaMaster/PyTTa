"""
Tests for pytta.psychoacoustics.

Each test pins one rule: a value that defines a unit, a reference value of a
standard, parity with SQAT (https://github.com/ggrecow/SQAT) on golden data,
or the rejection of an invalid input.

The SQAT golden data come from golden/make_loudness_sqat.m,
golden/make_roughness_sqat.m and golden/make_tonality_sqat.m (SQAT's
tonality is GPL-3.0 and serves only as an oracle; pytta's code for it is
written from the papers). Those scripts
store the parameters of short
synthetic signals; the helpers below rebuild the same samples (rounded to
float32, as SignalObj stores them) and check them against the sums stored
with the outputs.

SignalObj inspects the caller's frame (classes/_instanceinfo.py) and fails
when the caller's first local variable is an array, so the objects are built
inline in the tests, or in helpers whose first argument is a scalar.
"""
import os
import unittest

import numpy as np
import scipy.io as sio

import pytta
from pytta import psychoacoustics as pa

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')
P0 = 2e-5


def golden_cases(name):
    d = sio.loadmat(os.path.join(GOLDEN, name), squeeze_me=True,
                    struct_as_record=False)
    return list(np.atleast_1d(d['cases']))


def loudness_signal(p):
    """Rebuild the signal of one case of make_loudness_sqat.m."""
    n = int(round(p.dur * p.fs))
    t = np.arange(n) / p.fs
    if p.kind in ('tone', 'burst'):
        x = np.sqrt(2) * 2e-5 * 10 ** (p.L / 20) * np.sin(2 * np.pi * p.f * t)
        if p.kind == 'burst':
            x[~((t >= p.t0) & (t < p.t1))] = 0
    elif p.kind == 'ramp':
        T = n / p.fs
        x = (np.sqrt(2) * 2e-5 * 10 ** ((p.L + (p.L2 - p.L) * t / T) / 20)
             * np.sin(2 * np.pi * p.f * t))
    elif p.kind == 'multitone':
        K = 28
        a = np.sqrt(2) * 2e-5 * 10 ** (p.L / 20)
        x = np.zeros(n)
        for k in range(1, K + 1):
            fk = 1000 * 10 ** (((k - 1) - 16) / 10)
            x = x + a * np.sin(2 * np.pi * fk * t + np.pi * k * (k - 1) / K)
    return x.astype(np.float32)


def roughness_signal(p):
    """Rebuild the signal of one case of make_roughness_sqat.m."""
    n = int(round(p.dur * p.fs))
    t = np.arange(n) / p.fs
    A = np.sqrt(2) * 2e-5 * 10 ** (p.L / 20) / np.sqrt(1 + p.m ** 2 / 2)
    x = A * (1 + p.m * np.sin(2 * np.pi * p.fm * t)) * np.sin(2 * np.pi * p.fc * t)
    return x.astype(np.float32)


def tonality_signal(p):
    """Rebuild a tonality case of make_tonality_sqat.m."""
    t = np.arange(int(round(p.dur * p.fs))) / p.fs
    x = np.zeros(t.size)
    for f in np.atleast_1d(p.f):
        x = x + np.sqrt(2) * P0 * 10 ** (p.L / 20) * np.sin(2 * np.pi * f * t)
    if p.kind == 'tone+noise':
        K = int(p.f2 - p.f1 + 1)
        nz = np.zeros(t.size)
        for k in range(1, K + 1):
            nz = nz + np.sin(2 * np.pi * (p.f1 + k - 1) * t
                             + 2 * np.pi * np.mod(0.6180339887 * k ** 2, 1))
        x = x + nz * (P0 * 10 ** (p.Ln / 20) / np.sqrt(np.mean(nz ** 2)))
    return x.astype(np.float32)


def same_samples(tc, p, x):
    """The rebuilt float32 signal is the one MATLAB used."""
    x = x.astype(float)
    tc.assertLess(abs(np.sum(x ** 2) / p.sum_x2 - 1), 1e-12)
    tc.assertLess(abs(np.sum(x) - p.sum_x), 1e-12 * np.sum(np.abs(x)))


def band_noise(seed, f1, f2, level, fs=48000, dur=2.0):
    """Noise with a flat spectrum between f1 and f2 (Hz) at level dB SPL."""
    n = int(fs * dur)
    f = np.fft.rfftfreq(n, 1 / fs)
    rng = np.random.default_rng(seed)
    spec = np.exp(2j * np.pi * rng.random(f.size)) * ((f >= f1) & (f <= f2))
    x = np.fft.irfft(spec, n)
    return (x * P0 * 10 ** (level / 20) / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def at_loudness(target, x, fs=48000):
    """Scale x so that its stationary free-field loudness is target sone."""
    for _ in range(3):
        sig = pytta.SignalObj(x, 'time', fs)
        N = pa.loudness_zwicker(sig)['N']
        x = (x * 10 ** (10 * np.log2(target / N) / 20)).astype(np.float32)
    return x


class TestLoudness(unittest.TestCase):

    def test_1khz_tone_at_40_db_is_1_sone_and_40_phon(self):
        # Definition of the sone (ISO 532-1, 3.13 and Annex C). Zwicker's
        # tables reproduce it to 2.4e-5 sone (SQAT gives 0.999976).
        fs = 48000
        t = np.arange(fs) / fs
        sig = pytta.SignalObj(np.sqrt(2) * P0 * 100 * np.sin(2 * np.pi * 1000 * t),
                              'time', fs)
        out = pa.loudness_zwicker(sig)
        print(f'\n  N = {out["N"]:.6f} sone, LN = {out["LN"]:.4f} phon')
        self.assertAlmostEqual(out['N'], 1, delta=1e-4)
        self.assertAlmostEqual(out['LN'], 40, delta=1e-3)
        self.assertEqual(out['specific'].shape, (240,))
        np.testing.assert_allclose(out['bark'], np.arange(1, 241) / 10)

    def test_annex_b_stationary_reference_values(self):
        # ISO 532-1:2017 Annex B.2 and B.3, values of the test implementation
        # (test signals 1 to 4). Conformance (5.1): within 5 % or 0.1 sone.
        # Signal 1 is given as one-third-octave levels; 2 to 4 are tones,
        # synthesised here at the nominal level and analysed from 0.2 s on,
        # as B.1 prescribes (the filter transients at the start add up to
        # 1.6 % otherwise).
        levels = [-60, -60, 78, 79, 89, 72, 80, 89, 75, 87, 85, 79, 86, 80,
                  71, 70, 72, 71, 72, 74, 69, 65, 67, 77, 68, 58, 45, 30]
        out = pa._loudness_from_third_octave_levels(levels, 'free')
        N, LN = out['N'], out['LN']
        print(f'\n  signal 1: N = {N:.3f} (ref 83.296), LN = {LN:.3f} (ref 103.802)')
        self.assertLessEqual(abs(N - 83.296), max(0.05 * 83.296, 0.1))
        fs = 48000
        t = np.arange(fs) / fs
        for f, L, ref in [(250, 80, 14.655), (1000, 60, 4.019), (4000, 40, 1.549)]:
            sig = pytta.SignalObj(np.sqrt(2) * P0 * 10 ** (L / 20)
                                  * np.sin(2 * np.pi * f * t), 'time', fs)
            N = pa.loudness_zwicker(sig, time_skip=0.2)['N']
            print(f'  {f} Hz {L} dB: N = {N:.4f} sone, ref {ref}, diff {N - ref:+.4f}')
            self.assertLessEqual(abs(N - ref), max(0.05 * ref, 0.1))

    def test_matches_sqat_stationary(self):
        # Same float32 samples in both; what is left is double rounding in
        # the filters (and in the 44.1 to 48 kHz resampling, which follows
        # MATLAB's resample to 2e-15), far below 1e-9 sone.
        for p in golden_cases('loudness_sqat.mat'):
            if p.method != 1:
                continue
            x = loudness_signal(p)
            same_samples(self, p, x)
            sig = pytta.SignalObj(x, 'time', p.fs)
            out = pa.loudness_zwicker(sig, field=['free', 'diffuse'][p.field],
                                      time_skip=p.time_skip)
            dN = abs(out['N'] - p.N)
            dLN = abs(out['LN'] - p.LN)
            dspec = np.max(np.abs(out['specific'] - p.Nspec))
            print(f'\n  {p.name}: N = {out["N"]:.6f}, |dN| = {dN:.1e} sone, '
                  f'|dLN| = {dLN:.1e} phon, max |dN\'| = {dspec:.1e} sone/Bark')
            self.assertLess(dN, 1e-9)
            self.assertLess(dLN, 1e-9)
            self.assertLess(dspec, 1e-9)

    def test_matches_sqat_time_varying(self):
        # The nonlinear decay of step 8 picks its branch from comparisons
        # between successive values: core loudness that differ by 3e-14
        # sone (the rounding of the filter bank, MATLAB against scipy) come
        # out of it up to 5e-7 sone apart on the decay after a tone burst.
        # Tolerance 1e-6 sone (1e-5 phon).
        for p in golden_cases('loudness_sqat.mat'):
            if p.method != 2:
                continue
            x = loudness_signal(p)
            same_samples(self, p, x)
            sig = pytta.SignalObj(x, 'time', p.fs)
            out = pa.loudness_zwicker(sig, field=['free', 'diffuse'][p.field],
                                      method='time_varying',
                                      time_skip=p.time_skip)
            self.assertEqual(out['N'].shape, p.N.shape)
            self.assertEqual(out['specific'].shape, (p.N.size, 240))
            np.testing.assert_allclose(out['time'], p.time, rtol=0, atol=1e-12)
            dN = np.max(np.abs(out['N'] - p.N))
            dLN = np.max(np.abs(out['LN'] - p.LN))
            dspec = np.max(np.abs(out['specific'][p.Nspec_idx - 1] - p.Nspec_t))
            dstat = max(abs(out['N5'] - p.N5), abs(out['Nmax'] - p.Nmax))
            print(f'\n  {p.name}: Nmax = {out["Nmax"]:.5f}, max |dN(t)| = {dN:.1e}, '
                  f'|dLN| = {dLN:.1e}, |dN\'| = {dspec:.1e}, |dN5|,|dNmax| <= {dstat:.1e}')
            self.assertLess(dN, 1e-6)
            self.assertLess(dLN, 1e-5)
            self.assertLess(dspec, 1e-6)
            self.assertLess(dstat, 1e-6)

    def test_rejects_invalid_input(self):
        fs = 48000
        two = pytta.SignalObj(np.zeros((fs, 2)), 'time', fs)
        one = pytta.SignalObj(np.zeros(fs), 'time', fs)
        with self.assertRaises(ValueError):
            pa.loudness_zwicker(two)
        with self.assertRaises(ValueError):
            pa.loudness_zwicker(one, field='near')
        with self.assertRaises(ValueError):
            pa.loudness_zwicker(one, method='fast')
        with self.assertRaises(ValueError):
            pa.loudness_zwicker(one, time_skip=1.0)  # not shorter than the signal


class TestSharpness(unittest.TestCase):

    def test_din_reference_noise_is_1_acum(self):
        # DIN 45692, 3.1 and 6: narrowband noise 920 Hz to 1080 Hz at 60 dB
        # gives 1.00 acum. Tolerance: the DIN limit for measuring systems,
        # 5 % or 0.05 acum. This noise (seed 7) is not the one k was fitted on.
        x = band_noise(7, 920, 1080, 60)
        sig = pytta.SignalObj(x, 'time', 48000)
        S = pa.sharpness_din45692(sig)['S']
        print(f'\n  S = {S:.4f} acum, diff {S - 1:+.4f}')
        self.assertLessEqual(abs(S - 1), 0.05)

    def test_din_table_a2_a3_values(self):
        # DIN 45692 Tables A.2 (critical-band noises) and A.3 (broadband noise
        # from fu to 10 kHz), all at the loudness of the reference signal,
        # 4 sone. Tolerance: 5 % or 0.05 acum (DIN 45692, 6).
        cases = [  # (f1, f2, reference acum)
            (200, 300, 0.38), (920, 1080, 1.00), (3700, 4400, 2.82),
            (9500, 12000, 8.55),
            (250, 10000, 2.70), (1000, 10000, 3.05), (4000, 10000, 4.49),
            (8500, 10000, 7.46)]
        for f1, f2, ref in cases:
            x = at_loudness(4.0, band_noise(11, f1, f2, 60))
            sig = pytta.SignalObj(x, 'time', 48000)
            S = pa.sharpness_din45692(sig)['S']
            print(f'\n  {f1}-{f2} Hz: S = {S:.3f} acum, ref {ref}, diff {S - ref:+.3f}',
                  end='')
            self.assertLessEqual(abs(S - ref), max(0.05 * ref, 0.05))

    def test_time_varying_s50_of_reference_noise_is_1_acum(self):
        # DIN 45692, 6, note: S50 is the value to check with narrowband
        # noise, whose sharpness fluctuates. Same tolerance as above.
        x = band_noise(7, 920, 1080, 60, dur=1.0)
        sig = pytta.SignalObj(x, 'time', 48000)
        out = pa.sharpness_din45692(sig, method='time_varying', time_skip=0.2)
        print(f'\n  S50 = {out["S50"]:.4f} acum over {out["S"].size} frames')
        self.assertEqual(out['S'].shape, out['time'].shape)
        self.assertLessEqual(abs(out['S50'] - 1), 0.05)

    def test_silence_has_zero_sharpness(self):
        # N' = 0 everywhere leaves Formula (1) as 0/0; it is reported as 0.
        sig = pytta.SignalObj(np.zeros(48000), 'time', 48000)
        self.assertEqual(pa.sharpness_din45692(sig)['S'], 0)

    def test_rejects_two_channels(self):
        sig = pytta.SignalObj(np.zeros((48000, 2)), 'time', 48000)
        with self.assertRaises(ValueError):
            pa.sharpness_din45692(sig)


class TestRoughness(unittest.TestCase):

    def test_reference_am_tone_is_1_asper(self):
        # Definition of the asper: 1 kHz tone, 100 % amplitude modulated at
        # 70 Hz, 60 dB (rms of the modulated signal). SQAT calibrates on its
        # reference .wav file with a 7-digit constant and gives 1.000001 for
        # this synthetic version; 1e-4 asper leaves room for that only.
        fs = 48000
        t = np.arange(2 * fs) / fs
        A = np.sqrt(2) * P0 * 1000 / np.sqrt(1.5)
        x = A * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
        out = pa.roughness_daniel1997(pytta.SignalObj(x, 'time', fs), time_skip=0.5)
        print(f'\n  Rmean = {out["Rmean"]:.6f} asper, diff {out["Rmean"] - 1:+.1e}')
        self.assertAlmostEqual(out['Rmean'], 1, delta=1e-4)
        self.assertEqual(out['specific'].shape, (out['R'].size, 47))
        np.testing.assert_allclose(out['bark'], np.arange(1, 48) / 2)

    def test_matches_sqat(self):
        # The modulation depth and the correlation of channels whose
        # envelope sits at the FFT rounding floor depend on the rounding
        # pattern of the FFT (SQAT notes up to 5e-4 asper between two FFT
        # orderings in MATLAB); numpy's FFT is another ordering. Tolerance
        # 1e-3 asper on R(t) and on the specific roughness.
        for p in golden_cases('roughness_sqat.mat'):
            x = roughness_signal(p)
            same_samples(self, p, x)
            sig = pytta.SignalObj(x, 'time', p.fs)
            out = pa.roughness_daniel1997(sig, time_skip=p.time_skip)
            self.assertEqual(out['R'].shape, p.R.shape)
            np.testing.assert_allclose(out['time'], p.time, rtol=0, atol=1e-12)
            dR = np.max(np.abs(out['R'] - p.R))
            dspec = np.max(np.abs(out['specific'] - p.Rspec.T))
            dmean = abs(out['Rmean'] - p.Rmean)
            print(f'\n  {p.name}: Rmean = {out["Rmean"]:.6f}, max |dR(t)| = {dR:.1e}, '
                  f'max |dR\'| = {dspec:.1e}, |dRmean| = {dmean:.1e} asper')
            self.assertLess(dR, 1e-3)
            self.assertLess(dspec, 1e-3)
            self.assertLess(dmean, 1e-3)

    def test_rejects_invalid_input(self):
        fs = 48000
        two = pytta.SignalObj(np.zeros((fs, 2)), 'time', fs)
        with self.assertRaises(ValueError):
            pa.roughness_daniel1997(two)
        short = pytta.SignalObj(np.zeros(fs // 10), 'time', fs)
        with self.assertRaises(ValueError):
            pa.roughness_daniel1997(short)  # shorter than one 200 ms window


class TestTonality(unittest.TestCase):

    def test_reference_tone_is_1_tu(self):
        # Aures (1985), 3.1: a 1 kHz tone at 60 dB has a tonality of 1.
        fs = 48000
        t = np.arange(fs) / fs
        x = np.sqrt(2) * P0 * 1000 * np.sin(2 * np.pi * 1000 * t)
        out = pa.tonality_aures1985(pytta.SignalObj(x, 'time', fs))
        print(f'\n  Kmean = {out["Kmean"]:.6f} t.u., diff {out["Kmean"] - 1:+.1e}')
        self.assertAlmostEqual(out['Kmean'], 1, delta=1e-5)
        np.testing.assert_allclose(out['WGr'], 1, atol=1e-6)  # no noise left

    def test_level_excess_weighting(self):
        # A lone tone is masked by the threshold only, so its SPL excess is
        # L - LTH(f) (Terhardt et al. 1982, Eqs. 4 and 8). With w2 and WGr
        # unchanged, K(30 dB) / K(60 dB) = (w3'(30 - LTH) / w3'(60 - LTH))^0.29,
        # w3' = 1 - exp(-LX / 15 dB) (Aures 1985, Eqs. 10, 11 and 13).
        fs = 48000
        t = np.arange(fs) / fs
        x = np.sqrt(2) * P0 * 10 ** (30 / 20) * np.sin(2 * np.pi * 1000 * t)
        out = pa.tonality_aures1985(pytta.SignalObj(x, 'time', fs))
        lth = 3.64 - 6.5 * np.exp(-0.6 * (1 - 3.3) ** 2) + 1e-3
        w3 = 1 - np.exp(-(np.array([30, 60]) - lth) / 15)
        expected = (w3[0] / w3[1]) ** 0.29
        print(f'\n  K(30 dB) = {out["Kmean"]:.6f}, expected {expected:.6f}, '
              f'diff {out["Kmean"] - expected:+.1e}')
        self.assertAlmostEqual(out['Kmean'], expected, delta=1e-5)

    def test_matches_sqat(self):
        # Tones: 500 Hz and 4 kHz (w2), 30 dB (w3), two tones 1 and 1.2 kHz
        # (mutual masking in the SPL excess) agree to 1e-4 t.u. A 1 kHz tone
        # at 60 dB over a noise from 200 Hz to 5 kHz at 60 and 50 dB (masking
        # by noise and WGr): within 0.015 t.u. (the noise loudness comes
        # from the band levels of the spectrum here, from a resynthesised
        # signal in SQAT). SQAT's frames differ, so the means are compared.
        for p in golden_cases('tonality_sqat.mat'):
            x = tonality_signal(p)
            same_samples(self, p, x)
            out = pa.tonality_aures1985(pytta.SignalObj(x, 'time', p.fs))
            d = out['Kmean'] - p.Kmean
            print(f'\n  {p.name}: Kmean = {out["Kmean"]:.5f}, SQAT {p.Kmean:.5f}, diff {d:+.1e}')
            self.assertLess(abs(d), 0.015 if p.kind == 'tone+noise' else 1e-4)

    def test_rejects_invalid_input(self):
        fs = 48000
        one = pytta.SignalObj(np.zeros(fs), 'time', fs)
        with self.assertRaises(ValueError):
            pa.tonality_aures1985(one, field='near')
        two = pytta.SignalObj(np.zeros((fs, 2)), 'time', fs)
        with self.assertRaises(ValueError):
            pa.tonality_aures1985(two)
        short = pytta.SignalObj(np.zeros(fs // 10), 'time', fs)
        with self.assertRaises(ValueError):
            pa.tonality_aures1985(short)  # shorter than one 160 ms frame


if __name__ == '__main__':
    unittest.main()
