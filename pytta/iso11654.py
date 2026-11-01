# -*- coding: utf-8 -*-
"""
Rating of sound absorption, ISO 11654:1997: the practical sound absorption
coefficients of the octave bands from one-third octave values measured by
ISO 354, the weighted sound absorption coefficient alpha_w and the shape
indicators L, M and H.

>>> ap = pytta.iso11654.practical_absorption(alpha_third_octaves)
>>> alpha_w, shape = pytta.iso11654.weighted_absorption(ap)

"""

import numpy as np

__all__ = ['OCTAVES', 'REFERENCE', 'practical_absorption', 'weighted_absorption',
           'rating']

OCTAVES = np.array([125, 250, 500, 1000, 2000, 4000])
# Reference curve, 250 Hz to 4 kHz (Figure 1).
REFERENCE = np.array([0.80, 1.00, 1.00, 1.00, 0.90])


def practical_absorption(alpha):
    """
    Practical sound absorption coefficients alpha_p of the octave bands, one
    per three consecutive one-third octave values of alpha (4.1): their
    arithmetic mean to the second decimal, rounded in steps of 0.05 and
    limited to 1.00. Give the 18 one-third octave bands from 100 Hz to 5 kHz
    for the six octaves from 125 Hz to 4 kHz.
    """
    a = np.asarray(alpha, dtype=float)
    if a.size % 3:
        raise ValueError('Give three one-third octave values per octave.')
    hundredths = np.round(a.reshape(-1, 3).mean(axis=1) * 100)
    return np.minimum(np.round(hundredths / 5) * 5 / 100, 1.0)


def weighted_absorption(alpha_p):
    """
    Weighted sound absorption coefficient alpha_w and the shape indicators
    (4.2 and 4.3) from the practical coefficients of the octaves 250 Hz to
    4 kHz (5 values, or 6 starting at 125 Hz). The reference curve moves in
    steps of 0.05 towards the measured values until the sum of the
    unfavourable deviations, where the measured value is below the curve,
    is 0.10 or less; alpha_w is the shifted curve at 500 Hz. A practical
    coefficient 0.25 or more above the shifted curve adds L (250 Hz), M (500
    Hz or 1 kHz) or H (2 kHz or 4 kHz). Returns (alpha_w, indicators).
    """
    a = np.asarray(alpha_p, dtype=float)
    if a.size == 6:
        a = a[1:]
    if a.size != 5:
        raise ValueError('Give the octaves 250 Hz to 4 kHz (or 125 Hz to 4 kHz).')
    # Work in hundredths, so that the steps of 0.05 are exact.
    ap = np.round(a * 100).astype(int)
    ref = np.round(REFERENCE * 100).astype(int)
    shift = -100
    while np.sum(np.maximum(ref + shift + 5 - ap, 0)) <= 10:
        shift += 5
    curve = ref + shift
    excess = ap - curve >= 25
    shape = ''.join(letter for letter, bands in (('L', [0]), ('M', [1, 2]), ('H', [3, 4]))
                    if excess[bands].any())
    return curve[1] / 100, shape


def rating(alpha_w, shape):
    """The rating as ISO 11654 writes it, e.g. 0.60(M); two decimals."""
    return f'{alpha_w:.2f}' + (f'({shape})' if shape else '')
