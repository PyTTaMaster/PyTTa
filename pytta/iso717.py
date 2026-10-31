"""
PyTTa ISO 717-1 rating:
----------------------
Single-number quantity of airborne sound insulation, ISO 717-1:2013.

    >>> Rw, C, Ctr = pytta.iso717.airborne(R)

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
    shift = int(np.floor((tenths - reference).min() / 10))  # no deviation
    while np.maximum(reference + (shift + 1) * 10 - tenths, 0).sum() \
            <= tables['limit']:
        shift += 1
    Xw = tables['reference'][tables['at500']] + shift
    C = [int(_half_up(-10 * np.log10(np.sum(10 ** ((np.array(L) - tenths / 10)
                                                    / 10))), 1)) - Xw
         for L in tables['spectra']]
    return Xw, C[0], C[1]
