"""
PyTTa ISO 532-2:
----------------

Loudness of stationary sounds by the Moore-Glasberg method, ISO 532-2:2017,
for monaural, diotic and dichotic listening.

Available functions:

    >>> pytta.iso532_2.loudness(frequencies, levels, field='free')
    >>> pytta.iso532_2.third_octave_components(levels)
    >>> pytta.iso532_2.noise_band_components(f1, f2, spectrum_level)
    >>> pytta.iso532_2.loudness_level(N)

The sound is given as discrete sinusoidal components (Clause 5): tones as
they are (5.2), bands of white or pink noise (5.3) and one-third octave band
levels (5.5) through the two helper functions, which return the components
that the program of Annex C uses for them.

The method follows the text of the standard (Clauses 7 and 8): transfer to
the tympanic membrane (free field, diffuse field or at the eardrum) and to
the oval window, excitation patterns from rounded-exponential auditory
filters whose lower slope depends on the level, specific loudness at ERBn
numbers 1.8 to 38.9 Cam in steps of 0.1, and binaural inhibition. The
earphone correction files of 7.2.3 are not included; a flat response at the
tympanic membrane is field='eardrum'.

References
----------

ISO 532-2:2017. Acoustics - Methods for calculating loudness - Part 2:
Moore-Glasberg method.

Moore, B. C. J., & Glasberg, B. R. (2007). Modeling binaural loudness.
The Journal of the Acoustical Society of America, 121(3), 1604-1612.
"""

import numpy as np

__all__ = ['loudness', 'loudness_level', 'third_octave_components',
           'noise_band_components']

# Table 1: frequency (Hz), free field, diffuse field and middle ear (dB)
_TABLE1 = np.array([
    [20, 0.0, 0.0, -39.6], [25, 0.0, 0.0, -32.0], [31.5, 0.0, 0.0, -25.85],
    [40, 0.0, 0.0, -21.4], [50, 0.0, 0.0, -18.5], [63, 0.0, 0.0, -15.9],
    [80, 0.0, 0.0, -14.1], [100, 0.0, 0.0, -12.4], [125, 0.1, 0.1, -11.0],
    [160, 0.3, 0.3, -9.6], [200, 0.5, 0.4, -8.3], [250, 0.9, 0.5, -7.4],
    [315, 1.4, 1.0, -6.2], [400, 1.6, 1.6, -4.8], [500, 1.7, 1.7, -3.8],
    [630, 2.5, 2.2, -3.3], [750, 2.7, 2.7, -2.9], [800, 2.6, 2.9, -2.6],
    [1000, 2.6, 3.8, -2.6], [1250, 3.2, 5.3, -4.5], [1500, 5.2, 6.8, -5.4],
    [1600, 6.6, 7.2, -6.1], [2000, 12.0, 10.2, -8.5], [2500, 16.8, 14.9, -10.4],
    [3000, 15.3, 14.5, -7.3], [3150, 15.2, 14.4, -7.0], [4000, 14.2, 12.7, -6.6],
    [5000, 10.7, 10.8, -7.0], [6000, 7.1, 8.9, -9.2], [6300, 6.4, 8.7, -10.2],
    [8000, 1.8, 8.5, -12.2], [9000, -0.9, 6.2, -10.8], [10000, -1.6, 5.0, -10.1],
    [11200, 1.9, 4.5, -12.7], [12500, 4.9, 4.0, -15.0], [14000, 2.0, 3.3, -18.2],
    [15000, -2.0, 2.6, -23.8], [16000, 2.5, 2.0, -32.3], [20000, 2.5, 2.0, -45.5]])
_FIELDS = {'free': 1, 'diffuse': 2, 'eardrum': None}

# Table 2: centre frequency (Hz), excitation level at threshold (dB), 10 lg G (dB)
_TABLE2 = np.array([
    [50, 27.46, -24.31], [63, 23.45, -20.30], [80, 18.47, -15.32],
    [100, 15.13, -11.98], [125, 11.97, -8.82], [160, 9.34, -6.19],
    [200, 7.43, -4.28], [250, 5.75, -2.60], [315, 4.73, -1.58],
    [400, 3.92, -0.77], [500, 3.15, 0.0], [630, 3.15, 0.0], [750, 3.15, 0.0],
    [800, 3.15, 0.0], [1000, 3.15, 0.0]])
# Table 3: 10 lg G (dB), alpha
_TABLE3 = np.array([[-25.0, 0.26692], [-20.0, 0.25016], [-15.0, 0.23679],
                    [-10.0, 0.22228], [-5.0, 0.21055], [0.0, 0.20000]])
# Table 4: 10 lg G (dB) from -25 to 0 in steps of 0.5, A
_TABLE4_A = np.array([
    7.784, 7.667, 7.551, 7.435, 7.318, 7.210, 7.103, 6.996, 6.889, 6.782,
    6.675, 6.596, 6.517, 6.438, 6.360, 6.281, 6.202, 6.124, 6.047, 5.975,
    5.902, 5.823, 5.744, 5.665, 5.587, 5.510, 5.437, 5.364, 5.291, 5.218,
    5.145, 5.086, 5.027, 4.972, 4.918, 4.863, 4.808, 4.754, 4.699, 4.644,
    4.590, 4.542, 4.496, 4.451, 4.405, 4.359, 4.314, 4.268, 4.222, 4.177,
    4.131])
_TABLE4_G = np.arange(-25, 0.25, 0.5)
# Table 5: loudness level (phon), loudness (sone)
_TABLE5 = np.array([
    [0.0, 0.001], [2.2, 0.004], [4.0, 0.008], [5.0, 0.010], [7.5, 0.019],
    [10.0, 0.031], [15.0, 0.073], [20.0, 0.146], [25.0, 0.26], [30.0, 0.43],
    [35.0, 0.67], [40.0, 1.00], [45.0, 1.46], [50.0, 2.09], [55.0, 2.96],
    [60.0, 4.14], [65.0, 5.77], [70.0, 8.04], [75.0, 11.2], [80.0, 15.8],
    [85.0, 22.7], [90.0, 32.9], [95.0, 47.7], [100.0, 69.6], [105.0, 102.0],
    [110.0, 151.0], [115.0, 225.0], [120.0, 337.6]])

_C = 0.0617           # sone/Cam, Formula (7)
_D = 0.35             # 1/dB, Formula (5)
_B = 0.08             # 1/Cam, Formulae (10) and (11)
_THETA = 1.5978       # Formulae (12) and (13)
_CAM = np.round(np.arange(18, 390) / 10, 1)          # 1.8 to 38.9 Cam
_FC = (10 ** (_CAM / 21.366) - 1) / 0.004368         # Formula (6)


def _erb(f):
    return 24.673 * (0.004368 * f + 1)               # Formula (1)


def _interp(f, x, y):
    # 7.2.6: linear in dB over linear frequency, end values held
    return np.interp(f, x, y)


def _excitation(freqs, levels):
    """E/E0 at the 372 centre frequencies for components at the oval window."""
    I = 10 ** (levels / 10)
    p_c = 4 * freqs / _erb(freqs)
    # X: level of the roex filter centred on each component (Formulae (2) to (4))
    g = np.abs(freqs[None, :] - freqs[:, None]) / freqs[:, None]
    inside = np.where(freqs[None, :] < freqs[:, None], g <= 1, g <= 4)
    W = (1 + p_c[:, None] * g) * np.exp(-p_c[:, None] * g) * inside
    X = 10 * np.log10(W @ I)
    # auditory filters at the centre frequencies, lower slope from X (Formula (5))
    fc = _FC[:, None]
    pu = 4 * fc / _erb(fc)
    p1k = 4 * 1000 / _erb(1000)
    pl = pu - _D * (pu / p1k) * (X[None, :] - 51)
    g = np.abs(freqs[None, :] - fc) / fc
    below = freqs[None, :] < fc
    p = np.where(below, pl, pu)
    W = (1 + p * g) * np.exp(-p * g) * np.where(below, g <= 1, g <= 4)
    return W @ I


def _specific(E):
    """Specific loudness (sone/Cam) from E/E0, Formulae (7) to (9)."""
    LE_thr = _interp(_FC, _TABLE2[:, 0], _TABLE2[:, 1])
    G_dB = _interp(_FC, _TABLE2[:, 0], _TABLE2[:, 2])
    E_thr = 10 ** (LE_thr / 10)
    G = 10 ** (G_dB / 10)
    alpha = _interp(G_dB, _TABLE3[:, 0], _TABLE3[:, 1])
    A = _interp(G_dB, _TABLE4_G, _TABLE4_A)
    core = (G * E + A) ** alpha - A ** alpha
    N = _C * core
    low = E < E_thr
    N[low] = (_C * (2 * E[low] / (E[low] + E_thr[low])) ** 1.5 * core[low])
    high = E > 1e10
    N[high] = _C * (E[high] / 1.0707) ** 0.2
    return N


def _inhibition(NL, NR):
    """Binaural inhibition of 8.1, Formulae (10) to (13)."""
    d = np.round(np.arange(-180, 181) / 10, 1)
    weights = np.exp(-(_B * d) ** 2)
    n = NL.size

    def smooth(N):
        out = np.zeros(n)
        for w, k in zip(weights, np.round(d * 10).astype(int)):
            shifted = np.zeros(n)
            if k >= 0:
                shifted[k:] = N[:n - k] if k else N
            else:
                shifted[:k] = N[-k:]
            out += w * shifted
        return out

    SL, SR = smooth(NL) + 1e-13, smooth(NR) + 1e-13
    # With one ear silent the ratio reaches about 1e13: cosh overflows to
    # inf, so sech is 0, which is the limit the formulae need.
    with np.errstate(over='ignore'):
        inh_l = 2 / (1 + (1 / np.cosh(SR / SL)) ** _THETA)
        inh_r = 2 / (1 + (1 / np.cosh(SL / SR)) ** _THETA)
    return NL / inh_l, NR / inh_r


def _at_oval_window(freqs, levels, field):
    col = _FIELDS[field]
    outer = 0 if col is None else _interp(freqs, _TABLE1[:, 0], _TABLE1[:, col])
    return levels + outer + _interp(freqs, _TABLE1[:, 0], _TABLE1[:, 3])


def loudness_level(N):
    """
    Loudness level (phon) of a loudness N (sone) from Table 5, linear in
    lg N between its rows and held at 0 and 120 phon outside them.
    """
    return np.interp(np.log10(N), np.log10(_TABLE5[:, 1]), _TABLE5[:, 0])


def loudness(frequencies, levels, field='free', ears='both'):
    """
    Loudness of a stationary sound by the Moore-Glasberg method, ISO 532-2.

    Parameters
    ----------
    frequencies : array_like
        Frequencies of the sinusoidal components (Hz).
    levels : array_like
        Sound pressure levels of the components (dB re 20 uPa): (n,) for the
        same sound at both ears or at the one ear, or (n, 2) for the left and
        right ears; -inf leaves a component out of an ear.
    field : str, optional
        'free' (frontal incidence), 'diffuse' or 'eardrum' (levels at the
        tympanic membrane, which is also the case of earphones with a flat
        response there). Default 'free'.
    ears : str, optional
        'both' (binaural, the default) or 'one' (monaural, levels of shape
        (n,)).

    Returns
    -------
    dict
        'N': loudness (sone); 'LN': loudness level (phon, Table 5); 'cam':
        ERBn numbers (372,); 'specific': specific loudness (sone/Cam),
        (372,) monaural or (372, 2) left and right after inhibition;
        'N_ears': loudness of each ear (binaural only).
    """
    if field not in _FIELDS:
        raise ValueError(f'field must be one of {tuple(_FIELDS)}.')
    if ears not in ('both', 'one'):
        raise ValueError("ears must be 'both' or 'one'.")
    f = np.asarray(frequencies, dtype=float).ravel()
    L = np.asarray(levels, dtype=float)
    if L.ndim == 1:
        L = np.column_stack([L, L]) if ears == 'both' else L[:, None]
    elif ears == 'one' or L.shape[1] != 2:
        raise ValueError('levels of shape (n, 2) are for ears="both".')
    if L.shape[0] != f.size or np.any(f <= 0):
        raise ValueError('one positive frequency is needed per level.')
    def ear(levels):
        on = np.isfinite(levels)
        if not np.any(on):
            return np.zeros(_CAM.size)
        return _specific(_excitation(f[on], _at_oval_window(f[on], levels[on], field)))

    specific = np.column_stack([ear(L[:, e]) for e in range(L.shape[1])])
    if ears == 'one':
        N = specific[:, 0].sum() / 10
        return {'N': float(N), 'LN': float(loudness_level(N)), 'cam': _CAM.copy(),
                'specific': specific[:, 0]}
    left, right = _inhibition(specific[:, 0], specific[:, 1])
    N_ears = np.array([left.sum(), right.sum()]) / 10
    N = N_ears.sum()
    return {'N': float(N), 'LN': float(loudness_level(N)), 'cam': _CAM.copy(),
            'specific': np.column_stack([left, right]), 'N_ears': N_ears}


def noise_band_components(f1, f2, spectrum_level, pink=False, reference=1000.):
    """
    Components of a band of white or pink noise, 5.3: every 10 Hz, 10 dB above
    the spectrum level, for bands of 30 Hz or more, else every 1 Hz at the
    spectrum level. For pink noise the spectrum level is the one at
    reference (Hz) and falls 3 dB per octave.
    """
    step = 10. if f2 - f1 >= 30 else 1.
    freqs = np.arange(f1 + step / 2 if step == 10 else f1 + 1, f2 + 1e-9, step)
    L = spectrum_level + (10 if step == 10 else 0) * np.ones(freqs.size)
    if pink:
        L = L - 10 * np.log10(freqs / reference)
    return freqs, L


def third_octave_components(levels):
    """
    Components of 29 one-third octave band levels from 25 Hz to 16 kHz.

    As the program of Annex C (third.c), which produced the examples of
    Annex B: base-2 bands from 24.80314 Hz / 2^(1/6) in steps of 2^(1/3),
    10 components evenly spaced across each band up to 315 Hz and 20 from
    400 Hz on, each at LT - 10 lg(W / s) dB, W the width of the band and s
    the spacing, so that the components keep the band level. (The text of
    5.5 describes components every 10 Hz, or 1 Hz up to 125 Hz, with base-10
    bands; the program and Annex B use the scheme here.)

    Parameters
    ----------
    levels : array_like
        (29,) band levels (dB); -inf or nan leaves a band out.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Frequencies (Hz) and levels (dB) of the components.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.shape != (29,):
        raise ValueError('29 band levels are needed, 25 Hz to 16 kHz.')
    edges = 24.80314 / 2 ** (1 / 6) * 2 ** (np.arange(30) / 3)
    freqs, L = [], []
    for k, LT in enumerate(levels):
        if not np.isfinite(LT):
            continue
        n = 10 if k < 12 else 20
        W = edges[k + 1] - edges[k]
        freqs.append(edges[k] + (np.arange(n) + 0.5) * W / n)
        L.append(np.full(n, LT - 10 * np.log10(n)))
    return np.concatenate(freqs), np.concatenate(L)
