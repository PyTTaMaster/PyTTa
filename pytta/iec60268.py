"""
PyTTa IEC 60268-16 speech transmission index:
--------------------------------------------
Speech transmission index (STI) of IEC 60268-16:2011 from an impulse
response, by the indirect method of clause 6.

    >>> sti, mti, mtf = pytta.iec60268.sti(ir, levels, noise)

"""
import numpy as np

from pytta.classes import OctFilter

BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
MODULATION = np.array([0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8,
                       10, 12.5])                        # A.2.2
_ART = np.array([46, 27, 12, 6.5, 7.5, 8, 12])           # Table A.2
_WEIGHTS = {                                             # Table A.3
    False: (np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173]),
            np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])),
    True: (np.array([0, 0.117, 0.223, 0.216, 0.328, 0.250, 0.194]),
           np.array([0, 0.099, 0.066, 0.062, 0.025, 0.076])),
}


def _masking_db(L):
    """Auditory masking in dB from the level of the band below (Table A.1)."""
    return np.select([L < 63, L < 67, L < 100],
                     [0.5 * L - 65, 1.8 * L - 146.9, 0.5 * L - 59.8], -10)


def sti(ir, levels=None, noise=None, female=False):
    """
    Speech transmission index from an impulse response (IEC 60268-16:2011,
    clause 6 and Annex A).

    Parameters (default), (type):
    -----------------------------
        * ir (), (SignalObj): impulse response, one channel, at least 1.6 s
          long, with a sampling rate above 16 kHz;
        * levels (None), (7 values): speech level at the listener in the
          octave bands from 125 Hz to 8 kHz, in dB SPL. With None the
          auditory masking and the reception threshold (A.3) are left out,
          as for signals not measured in dB SPL;
        * noise (None), (7 values): background noise in the same bands, in
          dB SPL; needs levels;
        * female (False), (bool): female weighting of Table A.3.

    Return:
    -------
        * STI, limited to 1;
        * MTI: modulation transfer index of each octave band;
        * MTF: modulation transfer ratio, 14 modulation frequencies (rows)
          by 7 bands (columns), before the masking correction.

    The MTF comes from the Schroeder equation of 6.1 on each octave band,
    times 1 / (1 + 10**(-SNR / 10)) when noise is given. Masking (Table A.1,
    from the total level of the band below) and the reception threshold
    correct it as in A.5.3, and A.5.4 to A.5.6 give the STI.
    """
    for name, value in (('levels', levels), ('noise', noise)):
        if value is not None and len(value) != 7:
            raise ValueError(f'{name} must have 7 octave band values.')
    if noise is not None and levels is None:
        raise ValueError('noise needs the speech levels.')
    fs = ir.samplingRate
    bands = OctFilter(order=4, nthOct=1, samplingRate=fs, minFreq=125,
                      maxFreq=8000, refFreq=1000, base=10)
    h2 = bands.filter(ir)[0].timeSignal.astype(float) ** 2
    t = np.arange(h2.shape[0]) / fs
    mtf = np.abs(np.exp(-2j * np.pi * MODULATION[:, None] * t) @ h2) \
        / h2.sum(axis=0)
    m = mtf.copy()
    if levels is not None:
        signal = 10 ** (np.asarray(levels, dtype=float) / 10)
        total = signal.copy()
        if noise is not None:
            noiseI = 10 ** (np.asarray(noise, dtype=float) / 10)
            m = m / (1 + noiseI / signal)
            total = total + noiseI
        masking = np.r_[0, total[:-1] * 10 ** (_masking_db(
            10 * np.log10(total[:-1])) / 10)]
        m = m * signal / (signal + masking + 10 ** (_ART / 10))
    m = np.clip(m, 1e-12, 1 - 1e-12)
    snr = np.clip(10 * np.log10(m / (1 - m)), -15, 15)
    mti = np.mean((snr + 15) / 30, axis=0)
    alpha, beta = _WEIGHTS[female]
    value = np.sum(alpha * mti) - np.sum(beta * np.sqrt(mti[:-1] * mti[1:]))
    return min(float(value), 1.0), mti, mtf
