"""
PyTTa ECMA-418-1 Prominent Discrete Tones:
------------------------------------------

This module implements the two procedures of ECMA-418-1:2024 (3rd edition,
"Psychoacoustic metrics for ITT equipment, Part 1: prominent discrete tones")
for deciding whether a noise contains prominent discrete tones. The same
methods appear in ECMA-74, Annex D.

    * Tone-to-noise ratio, TNR (Clause 11 and Annex A): the power of the tone
      over the power of the masking noise in the critical band centred on it;

    * Prominence ratio, PR (Clause 12 and Annex B): the power of the critical
      band centred on the tone over the mean power of the two adjacent
      critical bands.

The calculations follow the mean-square formulation of Annexes A and B, which
the standard gives equal footing with the level formulation of Clauses 11 and
12. The spectrum is an averaged power spectrum (Welch method, Hann window,
linear averaging, Clause 7) with a frequency resolution finer than the one the
standard asks for.

Available functions:

    >>> pytta.ecma418.tone_to_noise_ratio(signal, tone_frequency=None)
    >>> pytta.ecma418.prominence_ratio(signal, tone_frequency=None)
    >>> pytta.ecma418.critical_band(frequency)

The audibility requirements (11.8, 12.8) and the screening tests near the
threshold of hearing (Clauses 8 and 9) need a listener or a calibrated level
and are left to the user: the returned 'prominent' flag only says whether the
ratio meets the criterion of 11.5 or 12.6.

For further information, check the function specific documentation.

"""

import numpy as np
from scipy import signal as ss

from pytta import SignalObj

# Discrete tone frequency range in which the criteria apply (4.1.1, 11.5, 12.6)
_F_MIN, _F_MAX = 89.1, 11200.0

# A spectral line is a tone candidate when it is the highest line within 15 %
# of the critical band centred on it and at least 10 dB above the median line
# of that critical band. The standard leaves the identification of tones to the
# operator (8.2 c); this rule only selects which lines are evaluated.
_CANDIDATE_DB = 10.0


def critical_band(frequency):
    """
    Lower and upper edges of the critical band centred on a frequency,
    ECMA-418-1, Clause 10.

    The bandwidth is Formula (2),

        Delta f_c = 25 + 75 * [1 + 1.4 * (f0 / 1000)**2]**0.69,

    with arithmetic edges up to 500 Hz (Formulae 4 and 5) and geometric edges
    above (Formulae 7 and 8).

    Parameters
    ----------
    frequency : float
        Centre frequency f0, in hertz.

    Returns
    -------
    f1, f2 : float
        Lower and upper band-edge frequencies, in hertz.

    """
    bandwidth = 25.0 + 75.0 * (1.0 + 1.4 * (frequency / 1000.0)**2)**0.69
    if frequency <= 500.0:
        f1 = frequency - bandwidth / 2
    else:
        f1 = (-bandwidth + np.sqrt(bandwidth**2 + 4 * frequency**2)) / 2
    return f1, f1 + bandwidth


def _adjacent_band_edges(frequency):
    """Lower edge of the lower band and upper edge of the upper band, Formulae (21), (22),
    Tables 2 and 3. Below 171,4 Hz the lower band starts at 20 Hz."""
    ft = frequency
    if ft <= 171.4:
        f1_lower = 20.0
    elif ft <= 1600.0:
        f1_lower = -149.5 + 1.001 * ft - 6.90e-5 * ft**2
    else:
        f1_lower = 6.8 + 0.806 * ft - 8.20e-6 * ft**2
    if ft <= 1600.0:
        f2_upper = 149.5 + 1.035 * ft + 7.70e-5 * ft**2
    else:
        f2_upper = 3.3 + 1.215 * ft + 2.16e-5 * ft**2
    return f1_lower, f2_upper


def _proximity_spacing(frequency):
    """Proximity spacing of two tones in a critical band, Formula (14); infinite from 1 kHz,
    where the criterion is always met."""
    if frequency >= 1000.0:
        return np.inf
    return 21.0 * 10**(1.2 * np.abs(np.log10(frequency / 212.0))**1.8)


def _tnr_criterion(frequency):
    """TNR criterion of 11.5, Formulae (12) and (13); NaN outside 89,1 Hz to 11 200 Hz."""
    if not _F_MIN <= frequency < _F_MAX:
        return np.nan
    return 8.0 + 8.33 * np.log10(1000.0 / frequency) if frequency < 1000.0 else 8.0


def _pr_criterion(frequency):
    """PR criterion of 12.6, Formulae (25) and (26); NaN outside 89,1 Hz to 11 200 Hz."""
    if not _F_MIN <= frequency < _F_MAX:
        return np.nan
    return 9.0 + 10.0 * np.log10(1000.0 / frequency) if frequency < 1000.0 else 9.0


def _power_spectrum(signal, resolution):
    """Welch power per spectral line (Hann, 50 % overlap, linear averaging), with a line
    spacing strictly finer than `resolution`."""
    if not isinstance(signal, SignalObj):
        raise TypeError("signal must be a pytta.SignalObj.")
    if signal.numChannels != 1:
        raise ValueError("signal must have exactly one channel.")
    x = np.asarray(signal.timeSignal[:, 0], dtype=float)
    fs = signal.samplingRate
    nperseg = 2**(int(np.floor(np.log2(fs / resolution))) + 1)
    if nperseg > x.size:
        raise ValueError(f"A frequency resolution below {resolution:.3g} Hz needs blocks "
                         f"of {nperseg / fs:.2f} s; the signal lasts {x.size / fs:.2f} s.")
    freq, psd = ss.welch(x, fs, window='hann', nperseg=nperseg,
                         noverlap=nperseg // 2, detrend=False)
    return freq, psd * (freq[1] - freq[0])


def _band(freq, f1, f2):
    """Lines in [f1, f2), so that contiguous bands never share a line."""
    return slice(*np.searchsorted(freq, [f1, f2]))


def _peak(freq, p, frequency):
    """Highest line within 15 % of the critical band around `frequency`."""
    f1, f2 = critical_band(frequency)
    lines = _band(freq, frequency - 0.075 * (f2 - f1), frequency + 0.075 * (f2 - f1))
    return lines.start + int(np.argmax(p[lines]))


def _candidates(freq, p, top):
    """Indices of the tone candidates between 89,1 Hz and 11 200 Hz whose bands end below `top`."""
    k = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    k = k[(freq[k] >= _F_MIN) & (freq[k] < _F_MAX)]
    found = []
    for i in k:
        f1, f2 = critical_band(freq[i])
        if f2 > top:
            break
        half = 0.075 * (f2 - f1)
        if p[i] < p[_band(freq, freq[i] - half, freq[i] + half)].max():
            continue
        if p[i] >= 10**(_CANDIDATE_DB / 10) * np.median(p[_band(freq, f1, f2)]):
            found.append(i)
    return np.array(found, dtype=int)


def _tone(freq, p, k):
    """Tone band end points and tone power above the line joining them, A.2, Formula (A.1)."""
    f1, f2 = critical_band(freq[k])
    window = _band(freq, freq[k] - 0.075 * (f2 - f1), freq[k] + 0.075 * (f2 - f1))
    lo, hi = window.start, window.stop - 1
    i1 = lo + int(np.argmin(p[lo:k + 1]))
    if i1 == lo:  # minimum on the window edge: take the next local minimum beyond it
        while i1 > 0 and p[i1 - 1] < p[i1]:
            i1 -= 1
    i2 = k + int(np.argmin(p[k:hi + 1]))
    if i2 == hi:
        while i2 < p.size - 1 and p[i2 + 1] < p[i2]:
            i2 += 1
    n = i2 - i1 + 1
    return i1, i2, p[i1:i2 + 1].sum() - n * (p[i1] + p[i2]) / 2


def _tnr(freq, p, k, others):
    """TNR of the tone at line k, A.2 to A.6. `others` are the other tone candidates; the
    strongest one in the critical band is the secondary tone. Returns the result and whether
    the secondary tone is a stronger proximate tone that absorbs this one."""
    f1, f2 = critical_band(freq[k])
    lines = _band(freq, f1, f2)
    i1, i2, x_tone = _tone(freq, p, k)
    x_tones = x_tone  # all tone power removed from the critical band
    inside = [j for j in others
              if lines.start <= j < lines.stop and not i1 <= j <= i2]
    absorbed = False
    if inside:
        j = max(inside, key=lambda m: p[m])
        x_sec = _tone(freq, p, j)[2]
        x_tones += x_sec  # the secondary tone always leaves the noise, (A.8)
        if abs(freq[j] - freq[k]) < _proximity_spacing(freq[k]):
            absorbed = x_sec > x_tone
            # perceived as one tone: the powers add, Formula (15) and the text of A.6
            # (Formula A.7 is printed with a minus sign)
            x_tone += x_sec
    x_noise = (p[lines].sum() - x_tones) \
        * (f2 - f1) / ((lines.stop - lines.start) * (freq[1] - freq[0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = 10 * np.log10(x_tone / x_noise)
    return _result(freq[k], ratio, _tnr_criterion(freq[k])), absorbed


def _pr(freq, p, k):
    """PR of the tone at line k, B.2 to B.5."""
    ft = freq[k]
    f1m, f2m = critical_band(ft)
    f1l, f2u = _adjacent_band_edges(ft)
    x_mid = p[_band(freq, f1m, f2m)].sum()
    x_low = p[_band(freq, f1l, f1m)].sum()
    x_up = p[_band(freq, f2m, f2u)].sum()
    if ft <= 171.4:  # truncated lower band referred to 100 Hz, Formula (B.8)
        x_low *= 100.0 / (f1m - f1l)
    ratio = 10 * np.log10(x_mid / (0.5 * (x_low + x_up)))  # Formulae (B.7), (B.8)
    return _result(ft, ratio, _pr_criterion(ft))


def _result(frequency, ratio, criterion):
    return {'frequency': float(frequency), 'ratio': float(ratio),
            'criterion': float(criterion), 'prominent': bool(ratio >= criterion)}


def tone_to_noise_ratio(signal, tone_frequency=None, resolution=None):
    """
    Tone-to-noise ratio of discrete tones, ECMA-418-1, Clause 11 and Annex A.

    The tone band is bounded by the minimal points on both sides of the tone
    peak within 15 % of the critical band (or the next local minimum beyond,
    when the minimum falls on that limit), and the tone power is the power
    above the line joining them (Formula A.1). The masking noise is the
    critical band power minus the tone, scaled from the bandwidth of the
    spectral lines to the critical bandwidth (Formula A.2). The strongest
    other tone in the critical band is the secondary tone of A.6: it is always
    removed from the noise and is added to the tone when closer than the
    proximity spacing of Formula (14) (always from 1 kHz).

    Parameters
    ----------
    signal : SignalObj
        One-channel sound pressure signal.
    tone_frequency : float, optional
        Frequency of the tone, in hertz. The tone peak is the highest spectral
        line within 15 % of the critical band around it. If None, the spectrum
        between 89,1 Hz and 11 200 Hz is searched for tone candidates: lines
        that are the highest within 15 % of their critical band and at least
        10 dB above the median line of it. A candidate merged into a stronger
        proximate tone is not reported on its own.
    resolution : float, optional
        Upper bound of the spectral line spacing, in hertz. Default 0,25 % of
        the tone frequency (of 89,1 Hz when searching), as Clause 7
        recommends for this method.

    Returns
    -------
    dict or list of dict
        One dict per tone, with keys 'frequency' (tone peak, Hz), 'ratio'
        (TNR, dB), 'criterion' (dB, Formulae 12 and 13; NaN outside 89,1 Hz to
        11 200 Hz) and 'prominent' (ratio >= criterion). A list sorted by
        frequency when tone_frequency is None.

    """
    f_ref = _F_MIN if tone_frequency is None else tone_frequency
    freq, p = _power_spectrum(signal, resolution or 0.0025 * f_ref)
    top = freq[-1]
    found = _candidates(freq, p, top)
    if tone_frequency is not None:
        if critical_band(tone_frequency)[1] > top:
            raise ValueError("The critical band exceeds half the sampling rate.")
        k = _peak(freq, p, tone_frequency)
        return _tnr(freq, p, k, [j for j in found if j != k])[0]
    results = []
    for k in found:
        res, absorbed = _tnr(freq, p, k, [j for j in found if j != k])
        if not absorbed:
            results.append(res)
    return results


def prominence_ratio(signal, tone_frequency=None, resolution=None):
    """
    Prominence ratio of discrete tones, ECMA-418-1, Clause 12 and Annex B.

    The ratio of the power in the critical band centred on the tone to the
    mean power of the lower and upper contiguous critical bands, with the
    outer edges from Formulae (21) and (22). For tones up to 171,4 Hz the
    lower band starts at 20 Hz and its power is referred to a 100 Hz
    bandwidth (Formula B.8).

    Parameters
    ----------
    signal : SignalObj
        One-channel sound pressure signal.
    tone_frequency : float, optional
        Frequency of the tone, in hertz. The tone peak is the highest spectral
        line within 15 % of the critical band around it. If None, the spectrum
        between 89,1 Hz and 11 200 Hz is searched for tone candidates as in
        tone_to_noise_ratio, and every candidate is reported.
    resolution : float, optional
        Upper bound of the spectral line spacing, in hertz. Default 1 % of the
        tone frequency (of 89,1 Hz when searching), Clause 7.

    Returns
    -------
    dict or list of dict
        One dict per tone, with keys 'frequency' (Hz), 'ratio' (PR, dB),
        'criterion' (dB, Formulae 25 and 26; NaN outside 89,1 Hz to
        11 200 Hz) and 'prominent' (ratio >= criterion). A list sorted by
        frequency when tone_frequency is None.

    """
    f_ref = _F_MIN if tone_frequency is None else tone_frequency
    freq, p = _power_spectrum(signal, resolution or 0.01 * f_ref)
    top = freq[-1]
    if tone_frequency is not None:
        if _adjacent_band_edges(tone_frequency)[1] > top:
            raise ValueError("The upper critical band exceeds half the sampling rate.")
        return _pr(freq, p, _peak(freq, p, tone_frequency))
    found = [k for k in _candidates(freq, p, top)
             if _adjacent_band_edges(freq[k])[1] <= top]
    return [_pr(freq, p, k) for k in found]
