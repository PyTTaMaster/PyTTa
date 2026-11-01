"""
Tests for the simulated measurement chain (SimulatedPlayRecMeasure and
SimulatedFRFMeasure, as itaMSTFdummy of the ITA-Toolbox) and for
multiple_sweep_excitation, multiple_sweep_irs and triggered_average of
pytta.functions. Each test measures a known system without a sound card.
"""
import unittest

import numpy as np

import pytta
from pytta.classes import SimulatedFRFMeasure, SimulatedPlayRecMeasure

FS = 48000


def signal(*samples, **kwargs):
    # A tuple as first local variable: SignalObj compares it with itself
    # while walking the stack, which an array cannot do before #59.
    return pytta.SignalObj(samples[0], 'time', FS, **kwargs)


def exponential_sweep(n, tail):
    t = np.arange(n) / FS
    L = n / FS / np.log(1000)
    x = 0.5 * np.sin(2 * np.pi * 20 * L * (np.exp(t / L) - 1))
    return signal(np.r_[x, np.zeros(tail)], freqMin=20, freqMax=20000)


def room(seed, n=2400, channels=2):
    rng = np.random.default_rng(seed)
    return 0.01 * np.exp(-np.arange(n) / 300)[:, None] * rng.standard_normal((n, channels))


def band_error(measured, true, n):
    # Largest relative difference of the spectra between 100 Hz and 10 kHz.
    f = np.fft.rfftfreq(n, 1 / FS)
    band = (f > 100) & (f < 10000)
    M, T = np.fft.rfft(measured, n, axis=0), np.fft.rfft(true, n, axis=0)
    return np.abs(M[band] - T[band]).max() / np.abs(T[band]).max()


class TestSimulatedMeasurement(unittest.TestCase):

    def test_frf_of_a_known_system(self):
        label = 'x'
        h = room(1)
        m = SimulatedFRFMeasure(system=h, excitation=exponential_sweep(2 ** 16, 2 ** 14),
                                inChannels=[1, 2], outChannels=[1], device=0)
        # The regularization limits the band, which spreads the response
        # in time, so the whole response is compared in the band.
        ir = np.asarray(m.run().IR.timeSignal, dtype=float)
        e = band_error(ir, h, ir.shape[0])
        print(f'\n  simulated FRF: {e:.1e}', end='')
        self.assertLess(e, 1e-3)

    def test_noise_and_quantization(self):
        # Silence gives the background noise level; a 16-bit chain adds the
        # quantization error of power step^2 / 12.
        label = 'x'
        quiet = signal(np.zeros(FS))
        rec = SimulatedPlayRecMeasure(system=room(2, channels=1), noise=-60., seed=3, excitation=quiet,
                                      inChannels=[1], outChannels=[1], device=0).run()
        level = 10 * np.log10(np.mean(np.asarray(rec.timeSignal, dtype=float) ** 2))
        tone = signal(0.5 * np.sin(2 * np.pi * 997 * np.arange(FS) / FS))
        ideal = SimulatedPlayRecMeasure(system=[[1.]], bits=53, excitation=tone,
                                        inChannels=[1], outChannels=[1], device=0).run()
        coarse = SimulatedPlayRecMeasure(system=[[1.]], bits=16, excitation=tone,
                                         inChannels=[1], outChannels=[1], device=0).run()
        e = np.asarray(coarse.timeSignal, dtype=float) - np.asarray(ideal.timeSignal, dtype=float)
        power = np.mean(e ** 2) / (2.0 ** -15) ** 2 * 12
        print(f'\n  noise {level:.2f} dB, quantization power {power:.3f} of step^2/12', end='')
        self.assertLess(abs(level + 60), 0.1)
        self.assertLess(abs(power - 1), 0.05)

    def test_multiple_sweeps_separate_the_sources(self):
        # Three sources, two microphones, sweeps 0.3 s apart, each response
        # cut 0.3 s long from 5 ms before its delay; without regularization
        # the deconvolution is exact, so the separation is.
        label = 'x'
        sweep = exponential_sweep(2 ** 15, 0)
        paths = [room(10 + k) for k in range(3)]
        x = pytta.multiple_sweep_excitation(sweep, 3, 0.3, tail=0.4)
        rec = SimulatedPlayRecMeasure(system=paths, excitation=x, inChannels=[1, 2],
                                      outChannels=[1, 2, 3], device=0).run()
        irs = pytta.multiple_sweep_irs(rec, sweep, 3, 0.3, 0.3, pretime=0.005,
                                       regularization=False)
        pre = int(0.005 * FS)
        e = max(band_error(np.asarray(ir.timeSignal, dtype=float),
                           np.pad(p, ((pre, 0), (0, 0))), int(0.3 * FS))
                for ir, p in zip(irs, paths))
        print(f'\n  multiple sweeps: {e:.1e}', end='')
        self.assertLess(e, 1e-3)

    def test_triggered_average(self):
        # Five hits of the same pulse in noise: five events, each starting
        # 1 ms before the first sample of the pulse above -20 dB, whose mean
        # is the pulse within the averaged noise.
        label = 'x'
        rng = np.random.default_rng(4)
        pulse = np.exp(-np.arange(960) / 100) * np.sin(2 * np.pi * 300 * np.arange(960) / FS)
        x = 1e-3 * rng.standard_normal(5 * FS)
        for k in range(5):
            x[k * FS + 1000:k * FS + 1960] += pulse
        mean, runs = pytta.triggered_average(signal(x), -20., 0.05, 0.001)
        i0 = np.argmax(np.abs(pulse) > 0.1)
        e = np.abs(np.asarray(mean.timeSignal[48:48 + 900, 0], dtype=float) - pulse[i0:i0 + 900]).max()
        print(f'\n  triggered average: {len(runs)} events, {e:.1e}', end='')
        self.assertEqual(len(runs), 5)
        self.assertLess(e, 3e-3)


if __name__ == '__main__':
    unittest.main()
