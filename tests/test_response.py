"""
Tests for pytta.response, measures of the response of a linear system:
resonance frequency and quality factor (from the -3 dB bandwidth and from
the decay), -3 dB cutoff frequencies, passband ripple, and the rise time,
overshoot and settling time of a step response. Each test pins one rule
on a system whose values are known exactly.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import scipy.signal as ss

import pytta
from pytta import response

FS = 48000


def resonator(f0, Q, seconds=2.0):
    """Impulse response of a second-order band-pass resonator."""
    b, a = ss.iirpeak(f0, Q, fs=FS)
    return ss.lfilter(b, a, np.eye(1, int(seconds * FS))[0])


class TestResponse(unittest.TestCase):

    def test_quality_factor_from_bandwidth_and_from_decay(self):
        # Both definitions give the Q of a second-order resonance.
        for f0, Q in ((1000.0, 10.0), (250.0, 30.0)):
            ir = pytta.SignalObj(resonator(f0, Q), 'time', FS)
            fr, Qb, bw = response.resonance(ir)
            Qd = response.q_from_decay(ir, f0)
            print(f'\n  f0 {f0}: found {fr:.2f} Hz, Q {Qb:.3f} (bandwidth), '
                  f'{Qd:.3f} (decay), expected {Q}')
            self.assertAlmostEqual(fr / f0, 1, delta=1e-3)
            self.assertAlmostEqual(Qb / Q, 1, delta=0.01)
            self.assertAlmostEqual(Qd / Q, 1, delta=0.02)

    def test_cutoffs_and_ripple_of_designed_filters(self):
        # A Butterworth band-pass is -3.01 dB at its design edges; a
        # Chebyshev type I low-pass ripples by rp in its passband.
        sos = ss.butter(4, [500, 2000], 'bandpass', fs=FS, output='sos')
        ir = pytta.SignalObj(ss.sosfilt(sos, np.eye(1, FS)[0]), 'time', FS)
        low, high = response.cutoff_frequencies(ir)
        self.assertAlmostEqual(low / 500, 1, delta=2e-3)
        self.assertAlmostEqual(high / 2000, 1, delta=2e-3)
        sos = ss.cheby1(6, 1.0, 4000, fs=FS, output='sos')
        ir = pytta.SignalObj(ss.sosfilt(sos, np.eye(1, FS)[0]), 'time', FS)
        self.assertAlmostEqual(response.passband_ripple(ir, 20, 3500), 1.0,
                               delta=0.02)

    def test_step_response_of_a_first_order_system(self):
        # y = 1 - exp(-t / tau): rise time (10 to 90 %) tau ln 9, no
        # overshoot, settling time within 2 % tau ln 50.
        tau = 0.01
        t = np.arange(FS // 2) / FS
        step = pytta.SignalObj(1 - np.exp(-t / tau), 'time', FS)
        rise, overshoot, settling = response.step_metrics(step)
        self.assertAlmostEqual(rise, tau * np.log(9), delta=2 / FS)
        self.assertAlmostEqual(overshoot, 0, delta=1e-9)
        self.assertAlmostEqual(settling, tau * np.log(50), delta=2 / FS)

    def test_overshoot_of_a_second_order_system(self):
        # Underdamped step response with damping ratio zeta: overshoot
        # exp(-pi zeta / sqrt(1 - zeta^2)).
        for zeta in (0.2, 0.5):
            wn = 2 * np.pi * 50
            b, a = ss.bilinear([wn ** 2], [1, 2 * zeta * wn, wn ** 2], FS)
            y = ss.lfilter(b, a, np.ones(FS))
            _, overshoot, _ = response.step_metrics(
                pytta.SignalObj(y, 'time', FS))
            expected = np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2))
            print(f'\n  zeta {zeta}: overshoot {overshoot:.4f} (expected '
                  f'{expected:.4f})')
            self.assertAlmostEqual(overshoot, expected, delta=2e-3)


if __name__ == '__main__':
    unittest.main()


class TestBifurcation(unittest.TestCase):

    def test_onset_extinction_and_hysteresis(self):
        # Subcritical behaviour: going up, the oscillation starts at
        # mu = 1.0; coming down, it persists until mu = 0.6. Below that only
        # noise 40 dB down. Hysteresis 0.4.
        up = np.round(np.arange(0.2, 1.41, 0.1), 2)
        down = up[::-1][1:]
        params = np.r_[up, down]
        rng = np.random.default_rng(0)
        t = np.arange(FS // 4) / FS
        recordings = []
        for i, mu in enumerate(params):
            on = mu >= 1.0 if i < up.size else mu >= 0.6
            amp = (0.5 + 0.2 * (mu - 0.6)) if on else 0.0
            x = amp * np.sin(2 * np.pi * 220 * t) \
                + 0.005 * rng.standard_normal(t.size)
            recordings.append(pytta.SignalObj(x, 'time', FS))
        result = response.bifurcation(params, recordings, freq=220)
        self.assertAlmostEqual(result['onset'], 1.0)
        self.assertAlmostEqual(result['extinction'], 0.6)
        self.assertAlmostEqual(result['hysteresis'], 0.4)
        self.assertEqual(len(result['amplitude']), params.size)
