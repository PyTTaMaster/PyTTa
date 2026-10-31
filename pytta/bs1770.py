"""
PyTTa ITU-R BS.1770 loudness and true peak:
------------------------------------------
Loudness of ITU-R BS.1770-5 (K-weighting, gated integrated loudness), the
momentary and short-term loudness of EBU Tech 3341, the loudness range of
EBU Tech 3342, and the true-peak level of BS.1770-5 Annex 2.

    >>> pytta.bs1770.loudness(signalObj)       # LUFS (LKFS)

Signals are taken in digital full scale (a sample of 1.0 is 0 dBFS), with
one channel per loudspeaker of the programme.
"""
import numpy as np
import scipy.signal as ss

# BS.1770-5, Annex 2: order 48, 4-phase FIR interpolating filter
_PHASES = np.array([
    [0.0017089843750, -0.0291748046875, -0.0189208984375, -0.0083007812500],
    [0.0109863281250, 0.0292968750000, 0.0330810546875, 0.0148925781250],
    [-0.0196533203125, -0.0517578125000, -0.0582275390625, -0.0266113281250],
    [0.0332031250000, 0.0891113281250, 0.1015625000000, 0.0476074218750],
    [-0.0594482421875, -0.1665039062500, -0.2003173828125, -0.1022949218750],
    [0.1373291015625, 0.4650878906250, 0.7797851562500, 0.9721679687500],
    [0.9721679687500, 0.7797851562500, 0.4650878906250, 0.1373291015625],
    [-0.1022949218750, -0.2003173828125, -0.1665039062500, -0.0594482421875],
    [0.0476074218750, 0.1015625000000, 0.0891113281250, 0.0332031250000],
    [-0.0266113281250, -0.0582275390625, -0.0517578125000, -0.0196533203125],
    [0.0148925781250, 0.0330810546875, 0.0292968750000, 0.0109863281250],
    [-0.0083007812500, -0.0189208984375, -0.0291748046875, 0.0017089843750],
])


def k_weighting(fs):
    """
    The two biquads of the K-weighting (stage 1 shelf, stage 2 high-pass),
    as ((b1, a1), (b2, a2)) at the sampling rate fs. The analog prototypes
    (shelf at 1681.97 Hz, +4.0 dB; high-pass at 38.135 Hz, as also used by
    libebur128) give Tables 1 and 2 of BS.1770-5 at 48 kHz, and the same
    response at other rates, as the Recommendation asks.
    """
    K = np.tan(np.pi * 1681.974450955533 / fs)
    Q = 0.7071752369554196
    Vh = 10 ** (3.999843853973347 / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K ** 2
    b1 = np.array([Vh + Vb * K / Q + K ** 2, 2 * (K ** 2 - Vh),
                   Vh - Vb * K / Q + K ** 2]) / a0
    a1 = np.array([a0, 2 * (K ** 2 - 1), 1 - K / Q + K ** 2]) / a0
    K = np.tan(np.pi * 38.13547087602444 / fs)
    Q = 0.5003270373238773
    a2 = np.array([1 + K / Q + K ** 2, 2 * (K ** 2 - 1),
                   1 - K / Q + K ** 2]) / (1 + K / Q + K ** 2)
    return (b1, a1), (np.array([1.0, -2.0, 1.0]), a2)


def _block_loudness(signalObj, weights, seconds, hop=0.1):
    """Loudness of each complete block of `seconds`, every `hop` seconds,
    and the time of the end of each block."""
    fs = signalObj.samplingRate
    x = np.asarray(signalObj.timeSignal, dtype=float)
    for b, a in k_weighting(fs):
        x = ss.lfilter(b, a, x, axis=0)
    weights = np.ones(x.shape[1]) if weights is None else np.asarray(weights)
    if weights.size != x.shape[1]:
        raise ValueError('weights must have one value per channel.')
    length, step = int(round(seconds * fs)), int(round(hop * fs))
    energy = np.vstack((np.zeros(x.shape[1]), np.cumsum(x ** 2, axis=0)))
    ends = np.arange(length, x.shape[0] + 1, step)
    z = (energy[ends] - energy[ends - length]) / length
    with np.errstate(divide='ignore'):
        return ends / fs, -0.691 + 10 * np.log10(z @ weights)


def _gated(z, threshold):
    """Mean of the block powers above the threshold, in LUFS."""
    return -0.691 + 10 * np.log10(np.mean(z[threshold]))


def loudness(signalObj, weights=None):
    """
    Integrated, gated loudness in LUFS (LKFS) of BS.1770-5: blocks of
    400 ms with 75 % overlap, an absolute gate at -70 LUFS and a relative
    gate 10 LU below the absolute-gated loudness.

    * weights (None): weight of each channel (Table 3); 1 for all by
      default; 1.41 for the surround channels of a 5.1 programme, 0 for LFE.
    """
    _, lj = _block_loudness(signalObj, weights, 0.4)
    power = 10 ** ((lj + 0.691) / 10)
    absolute = lj > -70
    relative = _gated(power, absolute) - 10
    return float(_gated(power, absolute & (lj > relative)))


def momentary(signalObj, weights=None):
    """Momentary loudness of EBU Tech 3341 (400 ms window, no gating),
    every 100 ms: (time of the end of each window in s, LUFS)."""
    return _block_loudness(signalObj, weights, 0.4)


def short_term(signalObj, weights=None):
    """Short-term loudness of EBU Tech 3341 (3 s window, no gating), every
    100 ms: (time of the end of each window in s, LUFS)."""
    return _block_loudness(signalObj, weights, 3.0)


def loudness_range(signalObj, weights=None):
    """
    Loudness range in LU of EBU Tech 3342: the difference between the 95th
    and 10th percentiles of the short-term loudness, after an absolute gate
    at -70 LUFS and a relative gate 20 LU below the mean of the absolute-
    gated values. Percentiles as in the MATLAB code of Tech 3342.
    """
    _, S = short_term(signalObj, weights)
    S = S[S >= -70]
    S = np.sort(S[S >= 10 * np.log10(np.mean(10 ** (S / 10))) - 20])
    n = S.size

    def pick(percent):              # MATLAB round: halves away from zero
        return S[int(np.floor((n - 1) * percent / 100 + 1 + 0.5)) - 1]
    return float(pick(95) - pick(10))


def true_peak(signalObj):
    """
    True-peak level in dBTP of BS.1770-5 Annex 2: the largest magnitude of
    the signal oversampled 4 times with the interpolating filter of the
    Annex, over all channels. Computed in floating point, so the 12.04 dB
    attenuation of the Annex is not needed.
    """
    x = np.asarray(signalObj.timeSignal, dtype=float)
    peak = np.max(np.abs(x))
    for phase in _PHASES.T:
        peak = max(peak, np.max(np.abs(ss.lfilter(phase, 1, x, axis=0))))
    return float(20 * np.log10(peak))
