"""
PyTTa ISO 3382-1 spatial measures:
---------------------------------
Early lateral energy fractions, late lateral sound level and interaural
cross correlation coefficient of ISO 3382-1:2009, Annexes A and B, from
impulse responses. Reverberation times, clarity, definition and centre time
are in pytta.RoomAnalysis.

    >>> JLF, JLFC = pytta.iso3382.lateral_fraction(omni, eight)
    >>> LJ = pytta.iso3382.late_lateral_level(omni, eight, p10)
    >>> IACC = pytta.iso3382.iacc(binaural, t1=0, t2=0.08)

The time origin is the start of the impulse response: the first sample
whose squared value is within 20 dB of the maximum, found on the
broadband responses and kept for every band.
"""
import numpy as np

from pytta.classes import OctFilter


def _start(*responses):
    energy = np.max([r ** 2 for r in responses], axis=0)
    return int(np.argmax(energy >= energy.max() / 100))


def _bands(signalObj, nthOct, minFreq, maxFreq):
    """Channels of signalObj as arrays, (samples, bands) per channel."""
    if nthOct is None:
        return [signalObj.timeSignal[:, [ch]].astype(float)
                for ch in range(signalObj.numChannels)]
    filt = OctFilter(order=4, nthOct=nthOct,
                     samplingRate=signalObj.samplingRate, minFreq=minFreq,
                     maxFreq=maxFreq, refFreq=1000, base=10)
    return [band.timeSignal.astype(float) for band in filt.filter(signalObj)]


def _span(t1, t2, start, fs, n):
    first = start + int(round(t1 * fs))
    last = n if t2 is None else min(n, start + int(round(t2 * fs)))
    return slice(first, last)


def lateral_fraction(omni, eight, nthOct=1, minFreq=125., maxFreq=4000.):
    """
    Early lateral energy fractions JLF (A.14) and JLFC (A.15), by band:

        JLF  = sum over 5 to 80 ms of pL**2 / sum over 0 to 80 ms of p**2
        JLFC = sum over 5 to 80 ms of pL p  / sum over 0 to 80 ms of p**2

    Parameters (default), (type):
    -----------------------------
        * omni (), (SignalObj): impulse response from an omnidirectional
          microphone, one channel;
        * eight (), (SignalObj): impulse response from a figure-of-eight
          microphone at the same point, its null towards the source;
        * nthOct (1), (int or None): bands per octave, or None for the
          broadband value;
        * minFreq, maxFreq (125, 4000), (float): band range, in Hz.

    Return: (JLF, JLFC), arrays with one value per band, or floats when
    nthOct is None.
    """
    fs = omni.samplingRate
    start = _start(omni.timeSignal[:, 0])
    p = _bands(omni, nthOct, minFreq, maxFreq)[0]
    pL = _bands(eight, nthOct, minFreq, maxFreq)[0]
    early = _span(0, 0.080, start, fs, len(p))
    lateral = _span(0.005, 0.080, start, fs, len(p))
    total = np.sum(p[early] ** 2, axis=0)
    JLF = np.sum(pL[lateral] ** 2, axis=0) / total
    JLFC = np.sum(pL[lateral] * p[lateral], axis=0) / total
    if nthOct is None:
        return float(JLF[0]), float(JLFC[0])
    return JLF, JLFC


def late_lateral_level(omni, eight, p10, distance=10., nthOct=1,
                       minFreq=125., maxFreq=4000.):
    """
    Late lateral sound level LJ (A.16), by band, in dB:

        LJ = 10 lg( integral from 80 ms to the end of pL**2
                    / integral over the whole response of p10**2 )

    Parameters (default), (type):
    -----------------------------
        * omni (), (SignalObj): impulse response from an omnidirectional
          microphone at the measurement point, one channel, used only to
          find the direct sound, the time origin;
        * eight (), (SignalObj): impulse response from a figure-of-eight
          microphone at the same point, its null towards the source;
        * p10 (), (SignalObj): impulse response of the same source in free
          field, from an omnidirectional microphone at the given distance,
          with the same calibration as eight;
        * distance (10), (float): distance of p10 in m; other than 10 m,
          LpE,10 = LpE,d + 20 lg(d/10) (A.4);
        * nthOct, minFreq, maxFreq: as in lateral_fraction.

    Return: array with one value per band, or a float when nthOct is None.
    """
    start = _start(omni.timeSignal[:, 0])
    pL = _bands(eight, nthOct, minFreq, maxFreq)[0]
    ref = _bands(p10, nthOct, minFreq, maxFreq)[0]
    late = _span(0.080, None, start, eight.samplingRate, len(pL))
    free = _span(0, None, _start(p10.timeSignal[:, 0]), p10.samplingRate,
                 len(ref))
    LJ = 10 * np.log10(np.sum(pL[late] ** 2, axis=0) / eight.samplingRate
                       / (np.sum(ref[free] ** 2, axis=0) / p10.samplingRate)) \
        - 20 * np.log10(distance / 10)
    return float(LJ[0]) if nthOct is None else LJ


def late_lateral_average(LJ):
    """
    Frequency-averaged late lateral sound level LJ,avg (A.17), the energy
    mean of LJ in the octave bands of 125 Hz, 250 Hz, 500 Hz and 1 000 Hz,
    in dB.
    """
    LJ = np.asarray(LJ, dtype=float)
    if LJ.shape != (4,):
        raise ValueError('LJ must have the 4 octave bands from 125 Hz to '
                         '1 000 Hz.')
    return float(10 * np.log10(np.mean(10 ** (LJ / 10))))


def iacc(binaural, t1=0., t2=None, nthOct=1, minFreq=125., maxFreq=4000.):
    """
    Interaural cross correlation coefficient (B.1, B.2) by band: the largest
    magnitude of the normalized cross correlation of the left and right ear
    responses over t1 to t2, for lags shorter than 1 ms.

    Parameters (default), (type):
    -----------------------------
        * binaural (), (SignalObj): left ear in channel 0, right in 1;
        * t1, t2 (0, None), (float): time window in s; None is the end of
          the response. 0 to 0.08 gives the early IACC, 0.08 to None the
          late one;
        * nthOct, minFreq, maxFreq: as in lateral_fraction.

    Return: array with one value per band, or a float when nthOct is None.
    """
    fs = binaural.samplingRate
    start = _start(binaural.timeSignal[:, 0], binaural.timeSignal[:, 1])
    left, right = _bands(binaural, nthOct, minFreq, maxFreq)
    window = _span(t1, t2, start, fs, len(left))
    pl, pr = left[window], right[window]
    maxLag = int(np.ceil(0.001 * fs)) - 1          # |tau| < 1 ms
    padded = np.pad(right, ((maxLag, maxLag), (0, 0)))
    offset = window.start + maxLag
    iacf = [np.sum(pl * padded[offset + tau:offset + tau + len(pl)], axis=0)
            for tau in range(-maxLag, maxLag + 1)]
    norm = np.sqrt(np.sum(pl ** 2, axis=0) * np.sum(pr ** 2, axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        values = np.max(np.abs(iacf), axis=0) / norm
    return float(values[0]) if nthOct is None else values
