"""Tests for pytta.ecma418 (ECMA-418-1:2024, tone-to-noise ratio and prominence ratio).

Each test pins one rule. The synthetic signals are tones of known amplitude in white
Gaussian noise of known one-sided power spectral density, so the band powers the
standard asks for are known analytically:

    tone power      A**2 / 2
    noise in a band N0 * bandwidth, with N0 = 2 * sigma**2 / fs
"""
import numpy as np
import pytest

import pytta
from pytta import ecma418

FS = 48000


def _signal(duration, tones=(), seed=0):
    """Unit-variance white noise plus sines given as (frequency, amplitude) pairs."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * FS)) / FS
    x = rng.standard_normal(t.size)
    for freq, amp in tones:
        x += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return pytta.SignalObj(signalArray=x, domain='time', samplingRate=FS)


N0 = 2.0 / FS  # one-sided PSD of unit-variance white noise


def test_band_formulas_match_standard_examples():
    """Clause 10 and 12.2 EXAMPLES: critical bandwidth and band edges; 11.6 EXAMPLE: proximity."""
    f1, f2 = ecma418.critical_band(1000.0)
    print('Delta f_c(1000) =', f2 - f1, ' f1,M =', f1, ' f2,M =', f2)
    assert f2 - f1 == pytest.approx(162.2, abs=0.05)
    assert f1 == pytest.approx(922.2, abs=0.05)
    assert f2 == pytest.approx(1084.4, abs=0.05)
    f1, f2 = ecma418.critical_band(500.0)
    print('Delta f_c(500) =', f2 - f1)
    assert f2 - f1 == pytest.approx(117.3, abs=0.05)
    assert (f1 + f2) / 2 == pytest.approx(500.0)  # arithmetic edges up to 500 Hz
    for fp, expected in ((150.0, 23.0), (850.0, 63.8)):
        got = ecma418._proximity_spacing(fp)
        print(f'Delta f_prox({fp}) = {got:.2f} (standard {expected})')
        assert got == pytest.approx(expected, abs=0.05)


def test_prominence_criteria_follow_formulas_12_13_25_26():
    """TNR: 8 dB from 1 kHz, 8 + 8.33 lg(1000/f) below; PR: 9 dB and 9 + 10 lg(1000/f)."""
    cases = {100.0: (16.33, 19.0), 500.0: (8.0 + 8.33 * np.log10(2), 9.0 + 10 * np.log10(2)),
             1000.0: (8.0, 9.0), 5000.0: (8.0, 9.0)}
    for freq, (tnr, pr) in cases.items():
        got = ecma418._tnr_criterion(freq), ecma418._pr_criterion(freq)
        print(f'{freq} Hz: TNR criterion {got[0]:.3f} (expected {tnr:.3f}),'
              f' PR criterion {got[1]:.3f} (expected {pr:.3f})')
        assert got == pytest.approx((tnr, pr), abs=1e-3)
    # outside 89,1 Hz <= f < 11 200 Hz the criteria do not apply (Clause 1, NOTE 1)
    assert np.isnan(ecma418._tnr_criterion(11200.0))
    assert np.isnan(ecma418._pr_criterion(80.0))


def test_tnr_of_tone_in_white_noise_is_analytic():
    """TNR = 10 lg[(A^2/2) / (N0 * Delta f_c)], with arithmetic (300 Hz) and geometric (2 kHz) edges."""
    for freq, amp in ((300.0, 0.3), (2000.0, 0.5)):
        res = ecma418.tone_to_noise_ratio(_signal(30.0, [(freq, amp)]), tone_frequency=freq)
        f1, f2 = ecma418.critical_band(freq)
        expected = 10 * np.log10(amp**2 / 2 / (N0 * (f2 - f1)))
        print(f'{freq} Hz: TNR {res["ratio"]:.3f} dB, analytic {expected:.3f} dB,'
              f' difference {res["ratio"] - expected:+.3f} dB')
        assert res['frequency'] == pytest.approx(freq, abs=1.0)
        assert res['ratio'] == pytest.approx(expected, abs=0.5)


def test_pr_of_tone_in_white_noise_is_analytic():
    """PR = 10 lg X_M - 10 lg[(X_L + X_U)/2], with the 100 Hz lower band rule at 150 Hz."""
    for freq, amp in ((150.0, 0.2), (1500.0, 0.3)):
        res = ecma418.prominence_ratio(_signal(30.0, [(freq, amp)]), tone_frequency=freq)
        f1m, f2m = ecma418.critical_band(freq)
        f1l, f2u = ecma418._adjacent_band_edges(freq)
        xm = amp**2 / 2 + N0 * (f2m - f1m)
        xl = N0 * (100.0 if freq <= 171.4 else f1m - f1l)
        xu = N0 * (f2u - f2m)
        expected = 10 * np.log10(xm / (0.5 * (xl + xu)))
        print(f'{freq} Hz: PR {res["ratio"]:.3f} dB, analytic {expected:.3f} dB,'
              f' difference {res["ratio"] - expected:+.3f} dB')
        assert res['ratio'] == pytest.approx(expected, abs=0.5)


def test_two_tones_in_a_critical_band_combine_only_when_proximate():
    """Clause 11.6: a secondary tone closer than Delta f_prox adds to the primary; it always leaves the noise."""
    fp, ap, a_s = 600.0, 0.3, 0.2  # Delta f_prox(600) = 40.7 Hz, band about 538 Hz to 662 Hz
    f1, f2 = ecma418.critical_band(fp)
    noise = N0 * (f2 - f1)
    for fs_, combined in ((fp + 25.0, True), (fp - 55.0, False)):
        res = ecma418.tone_to_noise_ratio(_signal(30.0, [(fp, ap), (fs_, a_s)]), tone_frequency=fp)
        tone = ap**2 / 2 + (a_s**2 / 2 if combined else 0.0)
        expected = 10 * np.log10(tone / noise)
        print(f'secondary at {fs_} Hz: TNR {res["ratio"]:.3f} dB, analytic {expected:.3f} dB,'
              f' difference {res["ratio"] - expected:+.3f} dB')
        assert res['ratio'] == pytest.approx(expected, abs=0.5)


def test_search_flags_the_tone_and_not_the_noise():
    """With tone_frequency None the spectrum is searched; pure noise yields no prominent tone."""
    noisy = _signal(30.0, seed=1)
    for method in (ecma418.tone_to_noise_ratio, ecma418.prominence_ratio):
        found = method(noisy)
        print(method.__name__, 'on noise: ratios', [round(r['ratio'], 2) for r in found])
        assert not any(r['prominent'] for r in found)
        at = method(noisy, tone_frequency=1000.0)
        print(method.__name__, 'on noise at 1 kHz:', at)
        assert not at['prominent']
        found = [r for r in method(_signal(30.0, [(2000.0, 0.5)], seed=2)) if r['prominent']]
        print(method.__name__, 'with a 2 kHz tone:', found)
        assert len(found) == 1 and found[0]['frequency'] == pytest.approx(2000.0, abs=1.0)


def test_rejects_invalid_input():
    """One-channel SignalObj only, long enough for one FFT block at the required resolution."""
    label = 'SignalObj inspects the caller frame; its first local must not be an array'
    two = pytta.SignalObj(signalArray=np.zeros((FS, 2)), domain='time', samplingRate=FS)
    with pytest.raises(ValueError):
        ecma418.tone_to_noise_ratio(two, tone_frequency=1000.0)
    with pytest.raises(ValueError):  # 0.25 % of 100 Hz needs blocks longer than 1 s
        ecma418.tone_to_noise_ratio(_signal(1.0), tone_frequency=100.0)
    with pytest.raises(TypeError):
        ecma418.prominence_ratio(np.zeros(FS), tone_frequency=1000.0)
