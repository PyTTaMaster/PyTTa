"""
PyTTa system response measures:
------------------------------
Measures of the response of a linear system, from its impulse response or
its step response: resonance frequency and quality factor (from the -3 dB
bandwidth or from the decay), -3 dB cutoff frequencies, passband ripple,
the rise time, overshoot and settling time of a step response, and the
onset and hysteresis of a self-sustained oscillation, the slope of a
decaying envelope, the inharmonicity of a tone and phase plane plots. They
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


def envelope_slope(signalObj, top=-5., bottom=-35.):
    """
    Slope, in dB per second, of the amplitude envelope (Hilbert) of a
    decaying transient, fitted on its decay from the maximum between `top`
    and `bottom` dB below it.
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
    return np.polyfit(t[fit], level[fit], 1)[0]


def q_from_decay(signalObj, f0, top=-5., bottom=-35.):
    """
    Quality factor from the decay of a single resonance at f0, in Hz:
    Q = pi f0 tau, with tau the time constant of the amplitude envelope,
    from envelope_slope between `top` and `bottom` dB below the maximum.
    """
    tau = -20 * np.log10(np.e) / envelope_slope(signalObj, top, bottom)
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


def bifurcation(params, recordings, freq=None, rise=20.):
    """
    Onset, extinction and hysteresis of a self-sustained oscillation along
    a sweep of a control parameter (e.g. gain or blowing pressure), first
    up and then down.

    Parameters (default), (type):
    -----------------------------
        * params (), (array-like): value of the control parameter at each
          recording, increasing up to its maximum and then decreasing;
        * recordings (), (list of SignalObj): one recording per value;
        * freq (None), (float): oscillation frequency in Hz; its amplitude
          is the magnitude of the spectral peak within 5 % of freq. With
          None, the rms of each recording is used;
        * rise (20), (float): how many dB above the quietest recording an
          amplitude has to be to count as oscillating.

    Return: dict with 'amplitude' (one value per recording), 'onset' (the
    first value going up that oscillates), 'extinction' (the last value
    going down that still oscillates) and 'hysteresis' (onset minus
    extinction). A missing onset or extinction is nan.
    """
    params = np.asarray(params, dtype=float)
    amplitude = []
    for rec in recordings:
        x = np.asarray(rec.timeSignal[:, 0], dtype=float)
        if freq is None:
            amplitude.append(np.sqrt(np.mean(x ** 2)))
            continue
        f = np.fft.rfftfreq(x.size, 1 / rec.samplingRate)
        X = 2 * np.abs(np.fft.rfft(x * np.hanning(x.size))) \
            / np.hanning(x.size).sum()
        amplitude.append(X[np.abs(f - freq) <= 0.05 * freq].max())
    amplitude = np.asarray(amplitude)
    oscillating = 20 * np.log10(amplitude / amplitude.min()) >= rise
    top = int(np.argmax(params))
    up = np.flatnonzero(oscillating[:top + 1])
    down = np.flatnonzero(oscillating[top:]) + top
    onset = params[up[0]] if up.size else np.nan
    extinction = params[down[-1]] if down.size else np.nan
    return {'amplitude': amplitude, 'onset': onset,
            'extinction': extinction, 'hysteresis': onset - extinction}


def inharmonicity(signalObj, f0, numPartials=10):
    """
    Inharmonicity of a tone with fundamental near f0, in Hz, after the
    stiff string model f_n = n f0 sqrt(1 + B n^2) (Fletcher, Normal
    vibration frequencies of a stiff piano string, J. Acoust. Soc. Am. 36,
    203, 1964). Each partial is the spectral peak (Hann window, zero padded
    8 times, parabolic interpolation) within f0 / 4 of where the partials
    found so far predict it.

    Return: dict with 'f0' and 'B' fitted by least squares on
    f_n^2 / n^2 = f0^2 + f0^2 B n^2, 'partials' (Hz) and 'cents', the
    deviation of each partial from n f0.
    """
    x = np.asarray(signalObj.timeSignal[:, 0], dtype=float)
    n_fft = 8 * x.size
    f = np.fft.rfftfreq(n_fft, 1 / signalObj.samplingRate)
    with np.errstate(divide='ignore'):
        L = 20 * np.log10(np.abs(np.fft.rfft(x * np.hanning(x.size), n_fft)))
    fit_f0, B, partials = f0, 0., []
    for n in range(1, numPartials + 1):
        guess = n * fit_f0 * np.sqrt(1 + B * n ** 2)
        band = np.flatnonzero(np.abs(f - guess) <= f0 / 4)
        k = band[np.argmax(L[band])]
        d = 0.5 * (L[k - 1] - L[k + 1]) / (L[k - 1] - 2 * L[k] + L[k + 1])
        partials.append(f[k] + d * (f[1] - f[0]))
        if n >= 2:
            m = np.arange(1, n + 1)
            c1, c0 = np.polyfit(m ** 2., np.asarray(partials) ** 2 / m ** 2, 1)
            fit_f0, B = np.sqrt(c0), c1 / c0
    partials = np.asarray(partials)
    cents = 1200 * np.log2(partials / (fit_f0 * np.arange(1, numPartials + 1)))
    return {'f0': fit_f0, 'B': B, 'partials': partials, 'cents': cents}


def plot_phase_plane(signalObj, quantity='position'):
    """
    Phase plane plots of a vibration: position against velocity, velocity
    against acceleration and acceleration against position. The signal is
    the `quantity` ('position', 'velocity' or 'acceleration'); the others
    are derived from it in the frequency domain (multiplying or dividing by
    j 2 pi f, DC set to zero), so it is taken as periodic. Returns the
    Figure.
    """
    import matplotlib.pyplot as plt
    order = ['position', 'velocity', 'acceleration']
    x = np.asarray(signalObj.timeSignal[:, 0], dtype=float)
    X = np.fft.rfft(x)
    jw = 2j * np.pi * np.fft.rfftfreq(x.size, 1 / signalObj.samplingRate)
    q = {}
    for i, name in enumerate(order):
        gain = np.zeros_like(jw)
        gain[1:] = jw[1:] ** (i - order.index(quantity))
        q[name] = x if name == quantity else np.fft.irfft(X * gain, x.size)
    units = {'position': 'Position', 'velocity': 'Velocity',
             'acceleration': 'Acceleration'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (a, b) in zip(axes, [('position', 'velocity'),
                                 ('velocity', 'acceleration'),
                                 ('acceleration', 'position')]):
        ax.plot(q[a], q[b])
        ax.set(xlabel=units[a], ylabel=units[b])
        ax.grid(True)
    fig.tight_layout()
    return fig
