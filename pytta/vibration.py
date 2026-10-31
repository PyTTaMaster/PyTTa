"""
PyTTa human vibration:
---------------------
Frequency weightings of whole-body vibration (ISO 2631-1, ISO 2631-2,
ISO 2631-4) and hand-arm vibration (ISO 5349-1), as specified for
instruments in ISO 8041, the evaluation values built on them (weighted
rms, maximum transient vibration value, vibration dose value), and
integration and differentiation of signals (acceleration, velocity,
displacement).

    >>> aw = pytta.vibration.rms(acceleration, 'Wk')      # m/s2

"""
import numpy as np
import scipy.signal as ss

from pytta.classes import SignalObj

_INF = np.inf
# ISO 8041 parameters: band limiting f1, f2 (Q = 1/sqrt(2)), a-v
# transition f3, f4, Q4, upward step f5, Q5, f6, Q6, gain K.
_PARAMETERS = {
    'Wb': (0.4, 100, 16, 16, 0.55, 2.5, 0.9, 4, 0.95, 1.024),
    'Wc': (0.4, 100, 8, 8, 0.63, _INF, 1, _INF, 1, 1),
    'Wd': (0.4, 100, 2, 2, 0.63, _INF, 1, _INF, 1, 1),
    'We': (0.4, 100, 1, 1, 0.63, _INF, 1, _INF, 1, 1),
    'Wf': (0.08, 0.63, _INF, 0.25, 0.86, 0.0625, 0.80, 0.10, 0.80, 1),
    'Wh': (10 ** 0.8, 10 ** 3.1, 100 / (2 * np.pi), 100 / (2 * np.pi), 0.64,
           _INF, 1, _INF, 1, 1),
    'Wj': (0.4, 100, _INF, _INF, 1, 3.75, 0.91, 5.32, 0.91, 1),
    'Wk': (0.4, 100, 12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91, 1),
    'Wm': (10 ** -0.1, 100, 1 / (0.028 * 2 * np.pi), 1 / (0.028 * 2 * np.pi),
           0.5, _INF, 1, _INF, 1, 1),
}


def weighting_filter(kind, fs):
    """
    Second-order sections of the frequency weighting `kind` ('Wb', 'Wc',
    'Wd', 'We', 'Wf', 'Wh', 'Wj', 'Wk' or 'Wm') at the sampling rate fs:
    the product of the band-limiting high-pass and low-pass (second-order
    Butterworth), the acceleration-velocity transition and the upward step
    of ISO 8041, through the bilinear transform.
    """
    if kind not in _PARAMETERS:
        raise ValueError(f'kind must be one of {sorted(_PARAMETERS)}.')
    f1, f2, f3, f4, Q4, f5, Q5, f6, Q6, K = _PARAMETERS[kind]
    w = lambda f: 2 * np.pi * f
    q = 1 / np.sqrt(2)
    # each factor as zeros, poles and gain, so that the low-frequency
    # poles keep their precision (a single polynomial does not)
    zeros, poles = [0.0, 0.0], []
    poles += list(np.roots([1, w(f1) / q, w(f1) ** 2]))          # Hh
    poles += list(np.roots([1, w(f2) / q, w(f2) ** 2]))          # Hl
    gain = K * w(f2) ** 2
    if np.isfinite(f4):                                          # Ht
        poles += list(np.roots([1, w(f4) / Q4, w(f4) ** 2]))
        gain *= w(f4) ** 2
        if np.isfinite(f3):
            zeros.append(-w(f3))
            gain /= w(f3)
    if np.isfinite(f5):                                          # Hs
        zeros += list(np.roots([1, w(f5) / Q5, w(f5) ** 2]))
        poles += list(np.roots([1, w(f6) / Q6, w(f6) ** 2]))
    return ss.zpk2sos(*ss.bilinear_zpk(zeros, poles, gain, fs))


def _weighted(signalObj, kind):
    x = np.asarray(signalObj.timeSignal, dtype=float)
    if kind is None:
        return x
    return ss.sosfilt(weighting_filter(kind, signalObj.samplingRate), x,
                      axis=0)


def rms(signalObj, kind=None):
    """Frequency-weighted rms value of each channel (no weighting with
    kind None)."""
    return np.sqrt(np.mean(_weighted(signalObj, kind) ** 2, axis=0))


def mtvv(signalObj, kind=None, tau=1.0):
    """Maximum transient vibration value: the maximum of the running rms
    with exponential averaging of time constant tau (1 s)."""
    a = _weighted(signalObj, kind)
    alpha = np.exp(-1 / (tau * signalObj.samplingRate))
    running = ss.lfilter([1 - alpha], [1, -alpha], a ** 2, axis=0)
    return np.sqrt(np.max(running, axis=0))


def vdv(signalObj, kind=None):
    """Fourth power vibration dose value, (integral of aw^4 dt)^(1/4)."""
    a = _weighted(signalObj, kind)
    return (np.sum(a ** 4, axis=0) / signalObj.samplingRate) ** 0.25


def _spectral(signalObj, power, fmin):
    x = np.asarray(signalObj.timeSignal, dtype=float)
    n = x.shape[0]
    f = np.fft.rfftfreq(n, 1 / signalObj.samplingRate)
    factor = np.zeros(f.size, dtype=complex)
    keep = f > max(fmin, 0)
    factor[keep] = (2j * np.pi * f[keep]) ** power
    if power >= 0:
        factor[0] = 0 if power > 0 else 1
    y = np.fft.irfft(np.fft.rfft(x, axis=0) * factor[:, None], n, axis=0)
    return SignalObj(y, 'time', samplingRate=signalObj.samplingRate,
                     signalType=signalObj.signalType)


def integrate(signalObj, times=1, fmin=0.):
    """Integrate `times` times in the frequency domain (divide by j 2 pi f),
    e.g. acceleration to velocity. DC and the bins up to fmin are set to 0,
    so the result has no drift; the signal is taken as periodic."""
    return _spectral(signalObj, -times, fmin)


def differentiate(signalObj, times=1):
    """Differentiate `times` times in the frequency domain (multiply by
    j 2 pi f), e.g. displacement to velocity. The signal is taken as
    periodic."""
    return _spectral(signalObj, times, 0.)
