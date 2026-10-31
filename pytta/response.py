"""
PyTTa system response measures:
------------------------------
Measures of the response of a linear system, from its impulse response or
its step response: resonance frequency and quality factor (from the -3 dB
bandwidth or from the decay), -3 dB cutoff frequencies, passband ripple,
and the rise time, overshoot and settling time of a step response. They
use the first channel of the SignalObj.

    >>> f0, Q, bandwidth = pytta.response.resonance(ir)

"""
import numpy as np
import scipy.signal as ss


def _magnitude(signalObj, padding=8):
    """Frequencies and magnitude spectrum of the first channel, zero padded
    `padding` times for finer interpolation."""
    x = np.asarray(signalObj.timeSignal[:, 0], dtype=float)
    n = padding * x.size
    return np.fft.rfftfreq(n, 1 / signalObj.samplingRate), \
        np.abs(np.fft.rfft(x, n))


def _crossing(f, y, level, start, step):
    """Frequency where y falls to `level` walking from `start` by `step`
    (+1 or -1), linearly interpolated; nan when it never does."""
    i = start
    while 0 <= i + step < y.size:
        if y[i + step] <= level:
            j = i + step
            return f[i] + (f[j] - f[i]) * (y[i] - level) / (y[i] - y[j])
        i += step
    return np.nan


def resonance(signalObj, fmin=0., fmax=None):
    """
    Resonance frequency, quality factor and -3 dB bandwidth of the highest
    peak of the magnitude response between fmin and fmax, in Hz:
    (f0, Q = f0 / bandwidth, bandwidth). The half-power points are
    interpolated on the squared magnitude.
    """
    f, H = _magnitude(signalObj)
    band = (f >= fmin) & (f <= (f[-1] if fmax is None else fmax))
    peak = np.flatnonzero(band)[np.argmax(H[band])]
    power = H ** 2
    low = _crossing(f, power, power[peak] / 2, peak, -1)
    high = _crossing(f, power, power[peak] / 2, peak, +1)
    return f[peak], f[peak] / (high - low), high - low


def q_from_decay(signalObj, f0, top=-5., bottom=-35.):
    """
    Quality factor from the decay of a single resonance at f0, in Hz:
    Q = pi f0 tau, with tau the time constant of the amplitude envelope
    (Hilbert), fitted on its decay between `top` and `bottom` dB below the
    maximum.
    """
    x = np.asarray(signalObj.timeSignal[:, 0], dtype=float)
    with np.errstate(divide='ignore'):
        level = 20 * np.log10(np.abs(ss.hilbert(x)))
    level -= level.max()
    start = np.argmax(level)
    fit = np.arange(x.size) >= start
    fit &= (level <= top) & (level >= bottom)
    fit &= np.arange(x.size) < start + np.argmax(level[start:] < bottom)
    t = np.arange(x.size) / signalObj.samplingRate
    slope = np.polyfit(t[fit], level[fit], 1)[0]           # dB per second
    tau = -20 * np.log10(np.e) / slope
    return np.pi * f0 * tau


def cutoff_frequencies(signalObj, reference=None):
    """
    Lower and upper -3 dB frequencies, in Hz, around the maximum of the
    magnitude response (re `reference` in dB, the maximum by default); nan
    for a side where the response does not fall by 3 dB.
    """
    f, H = _magnitude(signalObj)
    with np.errstate(divide='ignore'):
        L = 20 * np.log10(H)
    peak = np.argmax(L)
    level = (L[peak] if reference is None else reference) - 10 * np.log10(2)
    return _crossing(f, L, level, peak, -1), _crossing(f, L, level, peak, +1)


def passband_ripple(signalObj, fmin, fmax):
    """Peak-to-valley variation of the magnitude response between fmin
    and fmax, in dB."""
    f, H = _magnitude(signalObj)
    band = (f >= fmin) & (f <= fmax)
    L = 20 * np.log10(H[band])
    return L.max() - L.min()


def step_metrics(signalObj, final=None, tolerance=0.02):
    """
    Rise time (10 % to 90 % of the final value) in s, overshoot as a
    fraction of the final value, and settling time in s (from the start of
    the signal until it stays within `tolerance` of the final value) of a
    step response. The final value is the mean of the last 10 % of the
    signal unless given.
    """
    y = np.asarray(signalObj.timeSignal[:, 0], dtype=float)
    fs = signalObj.samplingRate
    if final is None:
        final = np.mean(y[-max(1, y.size // 10):])

    def first_time(fraction):
        reached = np.flatnonzero(y >= fraction * final)
        if reached.size == 0 or reached[0] == 0:
            return np.nan
        i = reached[0]
        return (i - 1 + (fraction * final - y[i - 1]) / (y[i] - y[i - 1])) / fs

    rise = first_time(0.9) - first_time(0.1)
    overshoot = max(0.0, (y.max() - final) / final)
    error = np.abs(y - final) - tolerance * abs(final)
    outside = np.flatnonzero(error > 0)
    if outside.size == 0:
        return rise, overshoot, 0.0
    i = outside[-1]
    if i == y.size - 1:                       # never settles
        return rise, overshoot, np.nan
    settling = (i + error[i] / (error[i] - error[i + 1])) / fs
    return rise, overshoot, settling
