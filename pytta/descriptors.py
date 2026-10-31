"""
PyTTa signal descriptors:
------------------------
Descriptors of a whole signal, one value per channel: crest factor and
kurtosis of the samples, and the centroid, spread, flatness and rolloff of
the power spectrum.

    >>> pytta.descriptors.crest_factor(signalObj, dB=True)

"""
import numpy as np


def _samples(signalObj):
    return np.asarray(signalObj.timeSignal, dtype=float)


def _power(signalObj):
    """Frequencies and power spectrum of each channel (one-sided, DC
    excluded)."""
    x = _samples(signalObj)
    f = np.fft.rfftfreq(x.shape[0], 1 / signalObj.samplingRate)
    return f[1:], np.abs(np.fft.rfft(x, axis=0)[1:]) ** 2


def crest_factor(signalObj, dB=False):
    """Peak magnitude over rms; in dB when dB is True."""
    x = _samples(signalObj)
    cf = np.max(np.abs(x), axis=0) / np.sqrt(np.mean(x ** 2, axis=0))
    return 20 * np.log10(cf) if dB else cf


def kurtosis(signalObj):
    """Pearson kurtosis of the samples: 3 for Gaussian noise, 1.5 for a
    sine, larger for impulsive signals."""
    x = _samples(signalObj)
    x = x - x.mean(axis=0)
    return np.mean(x ** 4, axis=0) / np.mean(x ** 2, axis=0) ** 2


def spectral_centroid(signalObj):
    """Power-weighted mean frequency, in Hz."""
    f, P = _power(signalObj)
    return f @ P / P.sum(axis=0)


def spectral_spread(signalObj):
    """Power-weighted standard deviation of frequency around the
    centroid, in Hz."""
    f, P = _power(signalObj)
    c = f @ P / P.sum(axis=0)
    return np.sqrt(np.sum((f[:, None] - c) ** 2 * P, axis=0)
                   / P.sum(axis=0))


def spectral_flatness(signalObj):
    """Geometric over arithmetic mean of the power spectrum: near 0 for a
    tone, exp(-gamma) = 0.56 for white noise."""
    f, P = _power(signalObj)
    with np.errstate(divide='ignore'):
        return np.exp(np.mean(np.log(P), axis=0)) / np.mean(P, axis=0)


def spectral_rolloff(signalObj, fraction=0.85):
    """Frequency below which `fraction` of the power lies, in Hz."""
    f, P = _power(signalObj)
    cumulative = np.cumsum(P, axis=0) / P.sum(axis=0)
    return f[np.argmax(cumulative >= fraction, axis=0)]
