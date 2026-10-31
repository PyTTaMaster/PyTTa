"""
PyTTa ISO 717 rating:
--------------------
Single-number quantities of airborne sound insulation, ISO 717-1:2013, and
of impact sound insulation, ISO 717-2.

    >>> Rw, C, Ctr = pytta.iso717.airborne(R)
    >>> Lnw, CI = pytta.iso717.impact(Ln)

"""
import numpy as np

# ISO 717-1:2013, Table 3 (reference values) and Table 4 (spectra No. 1 and
# No. 2 for C and Ctr), from 100 Hz to 3 150 Hz in thirds and from 125 Hz
# to 2 000 Hz in octaves.
_TABLES = {
    16: dict(reference=[33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56,
                        56, 56, 56],
             spectra=([-29, -26, -23, -21, -19, -17, -15, -13, -12, -11, -10,
                       -9, -9, -9, -9, -9],
                      [-20, -20, -18, -16, -15, -14, -13, -12, -11, -9, -8,
                       -9, -10, -11, -13, -15]),
             limit=320, at500=7),
    5: dict(reference=[36, 45, 52, 55, 56],
            spectra=([-21, -14, -8, -5, -4], [-14, -10, -7, -4, -6]),
            limit=100, at500=2),
}


def _half_up(x, step):
    """Round to a multiple of step, halves up (ISO 80000-1), clearing the
    binary error of the input first."""
    return np.floor(np.round(np.asarray(x, dtype=float) / step, 6) + 0.5)


def _shift(tenths, reference, limit):
    """Largest shift s, in dB, of the reference curve (in tenths of a dB)
    such that the deviations of reference + s above tenths sum to no more
    than limit."""
    shift = int(np.floor((tenths - reference).min() / 10))  # no deviation
    while np.maximum(reference + (shift + 1) * 10 - tenths, 0).sum() \
            <= limit:
        shift += 1
    return shift


def airborne(R):
    """
    Weighted sound reduction index and spectrum adaptation terms of
    ISO 717-1:2013, from R, R', Dn or DnT by band.

    Parameters:
    -----------
        * R (array-like):
            16 one-third-octave bands from 100 Hz to 3 150 Hz or 5 octave
            bands from 125 Hz to 2 000 Hz, in dB.

    Return:
    -------
        * (Xw, C, Ctr), integers in dB: Rw (or R'w, Dn,w, DnT,w) and the
          adaptation terms for pink noise and for urban traffic noise.

    R is rounded to 0.1 dB, halves up (4.4). The reference curve is shifted
    in steps of 1 dB until the sum of unfavourable deviations is as large as
    possible but not more than 32.0 dB in thirds or 10.0 dB in octaves, and
    Xw is its value at 500 Hz. C and Ctr follow Equations (1) and (2).
    Values are kept in tenths of a decibel as integers, so the limit is
    compared exactly.
    """
    tables = _TABLES.get(len(R))
    if tables is None:
        raise ValueError('R must have 16 one-third-octave or 5 octave bands.')
    tenths = _half_up(R, 0.1)
    reference = np.array(tables['reference']) * 10
    shift = _shift(tenths, reference, tables['limit'])
    Xw = tables['reference'][tables['at500']] + shift
    C = [int(_half_up(-10 * np.log10(np.sum(10 ** ((np.array(L) - tenths / 10)
                                                    / 10))), 1)) - Xw
         for L in tables['spectra']]
    return Xw, C[0], C[1]


# ISO 717-2 reference values for impact sound, from 100 Hz to 3 150 Hz in
# thirds and from 125 Hz to 2 000 Hz in octaves, as restated by Crocker
# (ed.), Handbook of Noise and Vibration Control (2007), chapter 106, Table 13
# and Figure 4. The octave rating is 5 dB below the curve at 500 Hz, and CI
# sums the bands up to 2 500 Hz (the first 15 thirds, all 5 octaves).
_IMPACT = {
    16: dict(reference=[62, 62, 62, 62, 62, 62, 61, 60, 59, 58, 57, 54, 51,
                        48, 45, 42],
             limit=320, at500=7, offset=0, upto2500=15),
    5: dict(reference=[67, 67, 65, 62, 49],
            limit=100, at500=2, offset=-5, upto2500=5),
}


def impact(Ln):
    """
    Weighted normalized impact sound pressure level and spectrum adaptation
    term of ISO 717-2, from Ln, L'n or L'nT by band.

    Parameters:
    -----------
        * Ln (array-like):
            16 one-third-octave bands from 100 Hz to 3 150 Hz, 5 octave
            bands from 125 Hz to 2 000 Hz, or 19 one-third-octave bands from
            50 Hz to 3 150 Hz, in dB.

    Return:
    -------
        * (Lnw, CI), integers in dB: Ln,w (or L'n,w, L'nT,w) and the
          adaptation term for walking noise; (Lnw, CI, CI50) with the
          19 bands, CI50 being CI,50-2500.

    Ln is rounded to 0.1 dB, halves up. An unfavourable deviation is Ln above
    the reference curve. The curve is shifted in steps of 1 dB until the sum
    of unfavourable deviations is as large as possible but not more than
    32.0 dB in thirds or 10.0 dB in octaves; Ln,w is its value at 500 Hz,
    less 5 dB in octaves. CI = Ln,sum - 15 - Ln,w, with Ln,sum the energetic
    sum of the bands from 100 Hz (125 Hz in octaves) to 2 500 Hz;
    CI,50-2500 sums from 50 Hz. Ln,w is rated on 100 Hz to 3 150 Hz only.
    """
    tenths = _half_up(Ln, 0.1)
    extended = len(tenths) == 19  # 50, 63 and 80 Hz come first
    low, tenths = (tenths[:3], tenths[3:]) if extended else ([], tenths)
    tables = _IMPACT.get(len(tenths))
    if tables is None:
        raise ValueError('Ln must have 16 or 19 one-third-octave or 5 octave '
                         'bands.')
    reference = np.array(tables['reference']) * 10
    # an impact deviation is the airborne one with both curves negated
    shift = -_shift(-tenths, -reference, tables['limit'])
    Lnw = tables['reference'][tables['at500']] + shift + tables['offset']
    bands = tenths[:tables['upto2500']]
    ranges = (bands, np.r_[low, bands]) if extended else (bands,)
    return (Lnw, *[int(_half_up(10 * np.log10(np.sum(10 ** (L / 100))), 1))
                   - 15 - Lnw for L in ranges])
