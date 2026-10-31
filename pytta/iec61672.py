"""
PyTTa IEC 61672-1 sound level meter:
-----------------------------------
Frequency weightings A, C and Z, time weightings F and S, and the levels
of a sound level meter of IEC 61672-1:2013, plus the percentile levels of a
time-weighted level.

    >>> pytta.iec61672.leq(signalObj, weighting='A')    # LAeq, dB

Levels are in dB re the dBRef of each channel (20 uPa for a channel in Pa)
and are returned with one value per channel.
"""
import numpy as np
import scipy.signal as ss

_TIME_CONSTANTS = {'F': 0.125, 'S': 1.0}            # 5.8.1


def _poles():
    """Pole frequencies f1 to f4 in Hz, from Annex E (E.2.4, E.3.4)."""
    fr, fL, fH, D = 1000.0, 10 ** 1.5, 10 ** 3.9, 1 / np.sqrt(2)
    b = (fr ** 2 + fL ** 2 * fH ** 2 / fr ** 2 - D * (fL ** 2 + fH ** 2)) \
        / (1 - D)
    c = fL ** 2 * fH ** 2
    f1 = np.sqrt((-b - np.sqrt(b ** 2 - 4 * c)) / 2)
    f4 = np.sqrt((-b + np.sqrt(b ** 2 - 4 * c)) / 2)
    fA = 10 ** 2.45
    return f1, (3 - 5 ** 0.5) / 2 * fA, (3 + 5 ** 0.5) / 2 * fA, f4


def weighting_filter(kind, fs):
    """
    Second-order sections of the frequency weighting 'A', 'C' or 'Z' at
    the sampling rate fs: the poles and zeros of Annex E through the
    bilinear transform, with 0 dB at 1 kHz. 'Z' returns None (no filter).
    """
    if kind not in ('A', 'C', 'Z'):
        raise ValueError("weighting must be 'A', 'C' or 'Z'.")
    if kind == 'Z':
        return None
    f1, f2, f3, f4 = _poles()
    poles = [f1, f1, f4, f4] + ([f2, f3] if kind == 'A' else [])
    zeros = [0.0] * (4 if kind == 'A' else 2)
    sos = ss.zpk2sos(*ss.bilinear_zpk(zeros, [-2 * np.pi * p for p in poles],
                                      1.0, fs))
    _, h = ss.sosfreqz(sos, worN=np.array([1000.0]), fs=fs)
    sos[0, :3] /= abs(h[0])
    return sos


def _weighted(signalObj, weighting):
    """Frequency-weighted samples (samples, channels) and the dB reference
    of each channel."""
    x = np.asarray(signalObj.timeSignal, dtype=float)
    sos = weighting_filter(weighting, signalObj.samplingRate)
    if sos is not None:
        x = ss.sosfilt(sos, x, axis=0)
    refs = np.array([signalObj.channels[ch].dBRef
                     for ch in signalObj.channels.mapping], dtype=float)
    return x, refs


def time_weighted_level(signalObj, weighting='A', time='F'):
    """
    Exponentially time-weighted sound level of 5.8 (F: 0.125 s, S: 1 s),
    at every sample: (time in s, level in dB of shape (samples, channels)).
    """
    if time not in _TIME_CONSTANTS:
        raise ValueError("time must be 'F' or 'S'.")
    x, refs = _weighted(signalObj, weighting)
    alpha = np.exp(-1 / (_TIME_CONSTANTS[time] * signalObj.samplingRate))
    meanSquare = ss.lfilter([1 - alpha], [1, -alpha], x ** 2, axis=0)
    with np.errstate(divide='ignore'):
        level = 10 * np.log10(meanSquare / refs ** 2)
    return np.arange(x.shape[0]) / signalObj.samplingRate, level


def leq(signalObj, weighting='A'):
    """Time-averaged (equivalent continuous) sound level over the signal."""
    x, refs = _weighted(signalObj, weighting)
    return 10 * np.log10(np.mean(x ** 2, axis=0) / refs ** 2)


def sel(signalObj, weighting='A'):
    """Sound exposure level, the time integral of the squared weighted
    signal re the reference squared times 1 s."""
    x, refs = _weighted(signalObj, weighting)
    return 10 * np.log10(np.sum(x ** 2, axis=0) / signalObj.samplingRate
                         / refs ** 2)


def lmax(signalObj, weighting='A', time='F'):
    """Maximum time-weighted sound level."""
    return np.max(time_weighted_level(signalObj, weighting, time)[1], axis=0)


def lpeak(signalObj, weighting='C'):
    """Peak sound level, from the largest magnitude of the weighted
    signal (C by default, as LCpeak)."""
    x, refs = _weighted(signalObj, weighting)
    return 20 * np.log10(np.max(np.abs(x), axis=0) / refs)


def percentile_levels(signalObj, percents=(10, 50, 90), weighting='A',
                      time='F'):
    """
    Percentile levels L_N: the time-weighted level exceeded during N % of
    the signal, for each N in percents. Returns an array of shape
    (len(percents), channels).
    """
    level = time_weighted_level(signalObj, weighting, time)[1]
    return np.percentile(level, 100 - np.asarray(percents, dtype=float),
                         axis=0)
