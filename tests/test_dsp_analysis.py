"""
Tests for pitch_detection, wiener_hopf and normxcorr2 of pytta.functions,
which follow ita_pitch_detection, ita_wiener_hopf_factorization and
ita_normxcorr2 of the ITA-Toolbox. Each test pins one rule on a signal
whose answer is known.
"""
import unittest

import numpy as np
import scipy.signal as ss

import pytta

FS = 44100


def signal(*samples):
    # A tuple as first local variable: SignalObj compares it with itself
    # while walking the stack, which an array cannot do before #59.
    return pytta.SignalObj(samples[0], 'time', FS)


def harmonic(f0, seconds, harmonics=range(1, 7)):
    t = np.arange(int(seconds * FS)) / FS
    return sum(np.sin(2 * np.pi * h * f0 * t) / h for h in harmonics)


class TestDSPAnalysis(unittest.TestCase):

    def test_pitch_of_a_harmonic_tone_and_a_missing_fundamental(self):
        # Within one bin of 2**16 points (0.67 Hz), also when the
        # fundamental itself is absent.
        df = FS / 2 ** 16
        for name, x in (('full', harmonic(220, 0.5)), ('no fundamental', harmonic(220, 0.5, range(2, 8)))):
            pitch, confidence, _ = pytta.pitch_detection(signal(x), 50, 2000)
            print(f'\n  pitch, {name}: {pitch[0, 0]:.2f} Hz, confidence {confidence[0, 0]:.2f}', end='')
            self.assertLessEqual(abs(pitch[0, 0] - 220), df)

    def test_pitch_track_follows_a_change(self):
        x = np.r_[harmonic(220, 0.5), harmonic(330, 0.5)]
        pitch, _, t = pytta.pitch_detection(signal(x), 50, 2000, blockSize=FS // 4)
        np.testing.assert_allclose(pitch[0], [220] * 2 + [330] * 2, atol=FS / 2 ** 16)
        np.testing.assert_allclose(t, [0, 0.25, 0.5, 0.75])

    def test_wiener_hopf_recovers_minimum_and_maximum_phase_parts(self):
        # x = h_min * h_max with h_max anticausal (at the end of the
        # buffer): the factors are h_min and h_max up to a gain, and their
        # spectra multiply back to the spectrum of x.
        label = 'x'
        n = 4096
        h_min = np.zeros(n)
        h_min[:3] = np.poly([0.5, -0.3])
        h_max = np.zeros(n)
        h_max[[0, -1, -2]] = np.poly([1 / 0.6, -1 / 0.4])[::-1] / np.poly([1 / 0.6, -1 / 0.4])[-1]
        x = np.fft.ifft(np.fft.fft(h_min) * np.fft.fft(h_max)).real
        plus, minus = pytta.wiener_hopf(signal(x))
        p = np.asarray(plus.timeSignal[:, 0], dtype=float)
        m = np.asarray(minus.timeSignal[:, 0], dtype=float)
        e_prod = np.abs(np.fft.fft(p) * np.fft.fft(m) - np.fft.fft(x)).max()
        e_min = np.abs(p / p[0] - h_min / h_min[0]).max()
        e_max = np.abs(m / m[0] - h_max / h_max[0]).max()
        print(f'\n  Wiener-Hopf: product {e_prod:.1e}, minimum phase {e_min:.1e}, maximum phase {e_max:.1e}', end='')
        self.assertLess(max(e_prod, e_min, e_max), 1e-5)

    def test_normxcorr2_against_its_definition(self):
        rng = np.random.default_rng(6)
        image = rng.standard_normal((12, 15))
        template = image[3:7, 5:10].copy()
        r = pytta.normxcorr2(template, image)
        self.assertEqual(r.shape, (15, 19))
        self.assertAlmostEqual(r[6, 9], 1.0, places=12)
        self.assertEqual(np.unravel_index(np.argmax(r), r.shape), (6, 9))
        # One lag by the definition, with the image zero-padded.
        pad = np.pad(image, ((3, 3), (4, 4)))
        patch = pad[2:6, 3:8]
        t = template - template.mean()
        q = patch - patch.mean()
        self.assertAlmostEqual(r[2, 3], np.sum(t * q) / np.sqrt(np.sum(t ** 2) * np.sum(q ** 2)), places=10)


if __name__ == '__main__':
    unittest.main()
