"""
Enhanced spectrogram on NumPy arrays.

Author: Sergio Aguirre, 2026. Released in PyTTa under the MIT license.
The same method is in the SQAT toolbox (GPL-3.0) as
gui/SQAT_GUI_enhanced_stft.m (https://github.com/ggrecow/SQAT).
The SignalObj interface is pytta.enhanced_spectrogram.
"""
import numpy as np
from scipy.ndimage import convolve1d

__all__ = ['enhanced_map']


def enhanced_map(x, fs, smoothing='readable', n_frames=2000, f_min=20,
                 bins_per_octave=192):
    """
    Enhanced spectrogram of one channel, as a map of mean square values.

    Nine Blackman-Harris windows from 8 to 512 ms (geometric spacing) are
    reassigned (Auger and Flandrin, 1995), the energy of each frame is held
    over the frame step of its window, each map is smoothed, and the maps are
    combined by geometric mean, so that only the energy that all the windows
    place at the same point of the time-frequency plane remains (the idea of
    Cheung and Lim, 1991, applied to reassigned maps).

    Returns t (M,) in s, f (F,) in Hz on a logarithmic grid, and the map
    (F, M) of mean square values in the unit of x squared: the cells of a
    column add up to the mean square value of that instant. See
    pytta.enhanced_spectrogram for the parameters.
    """
    if f_min >= fs / 2:
        raise ValueError(f'f_min ({f_min} Hz) must be below fs/2 '
                         f'({fs / 2} Hz).')
    if smoothing not in ('readable', 'sharp'):
        raise ValueError("smoothing must be 'readable' or 'sharp'.")
    x = np.asarray(x, dtype=float).ravel()
    numSamples = x.size
    durations = 8e-3 * 64 ** (np.arange(9) / 8)
    windows = 2 * _round_half_away(durations * fs / 2).astype(int)
    hop = max(1, int(_round_half_away(0.001 * fs)),
              int(np.ceil(numSamples / n_frames)))
    steps = np.maximum(1, _round_half_away(windows / 8).astype(int))
    numColumns = len(range(0, numSamples, hop))
    numRows = int(np.ceil(np.log2((fs / 2) / f_min) * bins_per_octave)) + 1
    f = f_min * 2 ** (np.arange(numRows) / bins_per_octave)
    cellWidth = f * (2 ** (1 / bins_per_octave) - 1)
    if smoothing == 'readable':
        sigmaTime = 0.004 * fs / hop
        sigmaFreq = 2 * (2 ** (1 / 96) - 1) * f / cellWidth
    else:
        sigmaTime = 0.001 * fs / hop
        sigmaFreq = 1 / cellWidth
    timeKernel = _gaussian_kernel(max(sigmaTime, 0.3))
    sigmaFreq = np.maximum(sigmaFreq, 0.3)
    logSum = np.zeros((numColumns, numRows))
    for N, step in zip(windows, steps):
        energy = np.concatenate(([0.], np.cumsum(np.concatenate(
            (np.zeros(N), x ** 2, np.zeros(N))))))
        frameMax = np.max(energy[N:] - energy[:-N])
        R = _reassigned_map(x, fs, N, step, hop, numColumns, numRows,
                            f_min, bins_per_octave, frameMax) / hop
        numHold = 2 * int(np.floor(step / hop / 2)) + 1
        if numHold > 1:
            R = convolve1d(R, np.ones(numHold) / numHold, axis=0,
                           mode='constant')
        R = convolve1d(R, timeKernel, axis=0, mode='constant')
        R = _smooth_frequency_rows(R, sigmaFreq)
        with np.errstate(divide='ignore'):
            logSum += np.log(R + 1e-6 * R.max())
    t = np.arange(numColumns) * hop / fs
    return t, f, np.exp(logSum / windows.size).T


def _round_half_away(x):
    """Round halves away from zero, as MATLAB does."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _gaussian_kernel(sigma):
    half = int(np.ceil(3 * sigma))
    kernel = np.exp(-np.arange(-half, half + 1) ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _smooth_frequency_rows(R, sigma):
    """Gaussian smoothing along the frequency axis (columns of R) with a
    width that changes with frequency: one kernel per geometric level."""
    level = _round_half_away(np.log(sigma) / np.log(1.25)).astype(int)
    out = R.copy()
    numFreqs = R.shape[1]
    for lev in np.unique(level):
        cols = np.flatnonzero(level == lev)
        kernel = _gaussian_kernel(1.25 ** lev)
        half = kernel.size // 2
        c1 = max(0, cols[0] - half)
        c2 = min(numFreqs, cols[-1] + half + 1)
        smoothed = convolve1d(R[:, c1:c2], kernel, axis=1, mode='constant')
        out[:, cols] = smoothed[:, cols - c1]
    return out


def _reassigned_map(x, fs, N, step, hop, numColumns, numRows, f_min,
                    bins_per_octave, frameMax):
    """Reassigned spectrogram of one unit-energy Blackman-Harris window of N
    samples and frame step `step`, accumulated on the output grid."""
    half = N // 2
    padded = np.concatenate((np.zeros(half), x, np.zeros(half + N)))
    n = np.arange(-half, half)
    phase = 2 * np.pi * (n + half) / N
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    w = a[0] - a[1] * np.cos(phase) + a[2] * np.cos(2 * phase) \
        - a[3] * np.cos(3 * phase)
    wd = a[1] * (2 * np.pi / N) * np.sin(phase) \
        - a[2] * (4 * np.pi / N) * np.sin(2 * phase) \
        + a[3] * (6 * np.pi / N) * np.sin(3 * phase)
    norm = 1 / np.linalg.norm(w)
    w, wd = w * norm, wd * norm
    wt = n * w
    freqBins = np.arange(half + 1) * fs / N
    starts = np.arange(0, x.size, step)
    R = np.zeros(numColumns * numRows)
    chunk = max(1, int(2e6 // N))
    for b in range(0, starts.size, chunk):
        s = starts[b:b + chunk]
        frames = padded[s[:, None] + np.arange(N)[None, :]]
        Xh = np.fft.rfft(frames * w, axis=1)
        Xt = np.fft.rfft(frames * wt, axis=1)
        Xd = np.fft.rfft(frames * wd, axis=1)
        power = np.abs(Xh) ** 2
        r = np.conj(Xh) / np.maximum(power, np.finfo(float).eps)
        tHat = s[:, None] + np.real(Xt * r)
        fHat = freqBins[None, :] - fs / (2 * np.pi) * np.imag(Xd * r)
        col = _round_half_away(tHat / hop)
        with np.errstate(divide='ignore'):
            row = _round_half_away(bins_per_octave * np.log2(
                np.maximum(fHat, np.finfo(float).eps) / f_min))
        ok = (power > 1e-10 * frameMax) & (col >= 0) & (col < numColumns) \
            & (row >= 0) & (row < numRows) & (fHat > 0)
        cells = (col[ok] + row[ok] * numColumns).astype(int)
        R += np.bincount(cells, weights=power[ok] * (2 * step / N),
                         minlength=R.size)
    return R.reshape(numRows, numColumns).T
