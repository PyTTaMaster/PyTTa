"""
PyTTa Ambisonics:
-----------------

Higher-order Ambisonics (HOA): real spherical harmonics, encoding, decoding
with five decoders, rotation and the conversions between the usual channel
conventions.

Available functions:

    >>> Y = pytta.ambisonics.real_sh(order, directions)
    >>> hoa = pytta.ambisonics.encode(signals, directions, order)
    >>> D, order = pytta.ambisonics.decoder(ls_directions, 'allrad')
    >>> ls = pytta.ambisonics.decode(hoa, D)
    >>> hoa_rot = pytta.ambisonics.rotate(hoa, yaw, pitch, roll)

Signals are (samples, channels) arrays. The HOA channels follow ACN ordering,
q = n^2 + n + m (0-based), with orthonormal real spherical harmonics (N3D)
and no Condon-Shortley phase. Directions are (points, 2) arrays of
[azimuth, elevation] in degrees, 0 deg in front and 90 deg to the left.
Conversions to SN3D, FuMa and SID are included, and bformat_fuma gives the
first-order FuMa B-format of Iceberg.

Origin of the code
------------------

This module is a port to Python of the MATLAB library Higher-Order-Ambisonics
by Archontis Politis (2015), https://github.com/polarch/Higher-Order-Ambisonics
(getRSH, encodeHOA_N3D, decodeHOA_N3D, encodeBformat, decodeBformat,
rotateHOA_N3D, rotateBformat, convert_N3D_SN3D, convert_N3D_FuMa,
convert_ACN_SID, getMaxREweights, getLayoutAmbisonicOrder, ambiDecoder,
allrad, csad), with the functions it uses from his Spherical-Harmonic-
Transform library, https://github.com/polarch/Spherical-Harmonic-Transform
(getSHrotMtx, euler2rotationMatrix, getTdesign and its t-designs, in
pytta/_tdesigns.py). Both are under the BSD 3-Clause licence reproduced
below. bformat_fuma follows native_bformat_fuma of Iceberg
(https://github.com/aguirreSL/Iceberg, BSD 3-Clause, Sergio Aguirre).

References
----------

Zotter, F., & Frank, M. (2012). All-round ambisonic panning and decoding.
Journal of the Audio Engineering Society, 60(10), 807-820.

Zotter, F., Pomberger, H., & Noisternig, M. (2012). Energy-preserving
ambisonic decoding. Acta Acustica united with Acustica, 98(1), 37-47.

Epain, N., Jin, C. T., & Zotter, F. (2014). Ambisonic decoding with
constant angular spread. Acta Acustica united with Acustica, 100(5), 928-936.

Ivanic, J., & Ruedenberg, K. (1996). Rotation matrices for real spherical
harmonics. Direct determination by recursion. Journal of Physical Chemistry,
100(15), 6342-6347 (with the corrections of 1998).

Licence of the ported code (both libraries)
-------------------------------------------

    Copyright (c) 2015, Archontis Politis
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import warnings
from math import factorial

import numpy as np
from scipy import signal as ss
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from scipy.special import eval_legendre, lpmv

from . import vbap
from ._tdesigns import T_DESIGNS


def _dirs(directions):
    return np.atleast_2d(np.asarray(directions, dtype=float))


def _unit(directions):
    d = np.radians(_dirs(directions))
    return np.column_stack([np.cos(d[:, 0]) * np.cos(d[:, 1]),
                            np.sin(d[:, 0]) * np.cos(d[:, 1]), np.sin(d[:, 1])])


def _directions(vectors):
    v = np.asarray(vectors, dtype=float)
    return np.degrees(np.column_stack([np.arctan2(v[:, 1], v[:, 0]),
                                       np.arctan2(v[:, 2], np.hypot(v[:, 0], v[:, 1]))]))


def _order(channels):
    order = int(round(np.sqrt(channels))) - 1
    if (order + 1)**2 != channels:
        raise ValueError('HOA signals need (order + 1)^2 channels.')
    return order


def real_sh(order, directions):
    """
    Orthonormal real spherical harmonics, ACN ordering, no Condon-Shortley
    phase (getRSH).

    Returns
    -------
    numpy.ndarray
        ((order + 1)^2, points).
    """
    d = np.radians(_dirs(directions))
    azi, x = d[:, 0], np.sin(d[:, 1])
    Y = np.zeros(((order + 1)**2, len(d)))
    Y[0] = 1 / np.sqrt(4 * np.pi)
    for n in range(1, order + 1):
        for m in range(-n, n + 1):
            a = abs(m)
            norm = np.sqrt((2 * n + 1) * factorial(n - a) / (4 * np.pi * factorial(n + a)))
            P = (-1)**a * lpmv(a, n, x)
            trig = (np.sqrt(2) * np.cos(m * azi) if m > 0 else
                    np.sqrt(2) * np.sin(a * azi) if m < 0 else 1.)
            Y[n * n + n + m] = norm * P * trig
    return Y


def t_design(degree):
    """
    Spherical t-design of degree 1 to 21 (getTdesign).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Unit vectors (points, 3) and their [azimuth, elevation] in degrees.
    """
    if not 1 <= degree <= 21:
        raise ValueError('t-designs are available for degrees 1 to 21.')
    vecs = np.array(T_DESIGNS[degree])
    return vecs, _directions(vecs)


def encode(signals, directions, order):
    """
    Encode source signals from their directions to HOA, N3D (encodeHOA_N3D).

    signals: (samples, sources); returns (samples, (order + 1)^2).
    """
    s = np.asarray(signals, dtype=float)
    s = s[:, None] if s.ndim == 1 else s
    return s @ real_sh(order, directions).T


def _filterbank(sig, cutoffs, fs):
    # applyFilterbank: FIR of order 100 (fir1, Hamming), zero phase by
    # dropping the 50 samples of group delay.
    n_band, half = len(cutoffs) + 1, 50
    pad = np.vstack([sig, np.zeros((half, sig.shape[1]))])
    bands = []
    for b in range(n_band):
        if b == 0:
            h = ss.firwin(101, cutoffs[0], fs=fs)
        elif b == n_band - 1:
            h = ss.firwin(101, cutoffs[-1], fs=fs, pass_zero=False)
        else:
            h = ss.firwin(101, [cutoffs[b - 1], cutoffs[b]], fs=fs, pass_zero=False)
        bands.append(ss.lfilter(h, 1, pad, axis=0)[half:])
    return bands


def _decode(sig, D, cutoffs, fs, channels):
    D = np.asarray(D, dtype=float)
    D = D[:, :, None] if D.ndim == 2 else D
    n_band = 1 if cutoffs is None else len(cutoffs) + 1
    if D.shape[2] != n_band:
        raise ValueError('One decoding matrix is needed for each frequency band.')
    if n_band == 1:
        return sig @ D[:, :channels, 0].T
    return sum(b @ D[:, :channels, k].T
               for k, b in enumerate(_filterbank(sig, cutoffs, fs)))


def decode(hoa, D, cutoffs=None, fs=None):
    """
    Decode HOA signals with a decoding matrix (decodeHOA_N3D).

    D: (loudspeakers, (order + 1)^2), or (loudspeakers, (order + 1)^2,
    bands) with cutoffs (bands - 1 frequencies in Hz) and fs for a decoding
    in frequency bands, split by FIR filters of order 100.
    """
    hoa = np.asarray(hoa, dtype=float)
    return _decode(hoa, D, cutoffs, fs, hoa.shape[1])


def encode_bformat(signals, directions):
    """First-order FuMa B-format (W/sqrt(2), X, Y, Z) of sources (encodeBformat)."""
    s = np.asarray(signals, dtype=float)
    s = s[:, None] if s.ndim == 1 else s
    u = _unit(directions)
    return s @ np.column_stack([np.ones(len(u)) / np.sqrt(2), u])


def decode_bformat(bformat, D, cutoffs=None, fs=None):
    """Decode FuMa B-format with the first 4 columns of D (decodeBformat)."""
    return _decode(fuma_to_n3d(bformat), D, cutoffs, fs, 4)


def n3d_to_sn3d(sig):
    """N3D to SN3D normalization (convert_N3D_SN3D, 'n2sn')."""
    sig = np.asarray(sig, dtype=float)
    n = np.floor(np.sqrt(np.arange(sig.shape[1])))
    return sig / np.sqrt(2 * n + 1)


def sn3d_to_n3d(sig):
    """SN3D to N3D normalization (convert_N3D_SN3D, 'sn2n')."""
    sig = np.asarray(sig, dtype=float)
    n = np.floor(np.sqrt(np.arange(sig.shape[1])))
    return sig * np.sqrt(2 * n + 1)


_FUMA2ACN = np.array([1, 3, 4, 2, 9, 7, 5, 6, 8, 16, 14, 12, 10, 11, 13, 15]) - 1
_NORM_FUMA2N3D = np.array([np.sqrt(2), np.sqrt(3), np.sqrt(3), np.sqrt(3),
                           np.sqrt(15) / 2, np.sqrt(15) / 2, np.sqrt(5), np.sqrt(15) / 2,
                           np.sqrt(15) / 2, np.sqrt(35 / 8), np.sqrt(35) / 3,
                           np.sqrt(224 / 45), np.sqrt(7), np.sqrt(224 / 45),
                           np.sqrt(35) / 3, np.sqrt(35 / 8)])


def _fuma_order(sig):
    sig = np.asarray(sig, dtype=float)
    order = _order(sig.shape[1])
    if order > 3:
        warnings.warn('FuMa is defined up to third order; keeping the first 16 channels.')
        sig, order = sig[:, :16], 3
    return sig, (order + 1)**2


def fuma_to_n3d(sig):
    """FuMa (up to third order) to N3D, ACN (convert_N3D_FuMa, 'fuma2n')."""
    sig, nsh = _fuma_order(sig)
    return sig[:, _FUMA2ACN[:nsh]] * _NORM_FUMA2N3D[:nsh]


def n3d_to_fuma(sig):
    """N3D, ACN to FuMa (up to third order) (convert_N3D_FuMa, 'n2fuma')."""
    sig, nsh = _fuma_order(sig)
    acn2fuma = np.argsort(_FUMA2ACN)
    norm = (1 / _NORM_FUMA2N3D)[acn2fuma]
    return sig[:, acn2fuma[:nsh]] * norm[:nsh]


def _sid_index(order):
    idx = [0]
    for n in range(1, order + 1):
        ms = [m for k in range(1, n + 1) for m in (n - k + 1, k - n - 1)] + [0]
        idx += [n * n + n + m for m in ms]
    return np.array(idx)


def acn_to_sid(sig):
    """ACN to SID channel ordering (convert_ACN_SID, 'acn2sid')."""
    sig = np.asarray(sig)
    return sig[:, _sid_index(_order(sig.shape[1]))]


def sid_to_acn(sig):
    """SID to ACN channel ordering (convert_ACN_SID, 'sid2acn')."""
    sig = np.asarray(sig)
    return sig[:, np.argsort(_sid_index(_order(sig.shape[1])))]


def bformat_fuma(sig, convention):
    """
    First-order FuMa B-format (W, X, Y, Z) from full-sphere Ambisonics of
    any order, as Iceberg's native_bformat_fuma.

    convention: 'fuma', 'ambix' (ACN, SN3D) or 'n3d' (ACN, N3D). The first
    four channels hold the first order in all three, so the conversion of
    those four is exact.

    Returns
    -------
    (numpy.ndarray, int)
        (samples, 4) FuMa B-format and the order of the input.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 2 or sig.shape[1] < 4:
        raise ValueError('Ambisonics B-format needs (order + 1)^2 channels, 4 or more.')
    order = _order(sig.shape[1])
    c = convention.lower()
    if c == 'fuma':
        return sig[:, :4], order
    if c == 'ambix':
        return np.column_stack([sig[:, 0] / np.sqrt(2), sig[:, 3], sig[:, 1], sig[:, 2]]), order
    if c == 'n3d':
        return np.column_stack([sig[:, 0] / np.sqrt(2), sig[:, [3, 1, 2]] / np.sqrt(3)]), order
    raise ValueError("convention must be 'fuma', 'ambix' or 'n3d'.")


def max_re_weights(order):
    """
    Max-rE weights a_n = P_n(cos(137.9 deg / (order + 1.51))), repeated per
    channel (getMaxREweights).
    """
    x = np.cos(np.radians(137.9) / (order + 1.51))
    return np.concatenate([np.full(2 * n + 1, eval_legendre(n, x))
                           for n in range(order + 1)])


def layout_order(ls_directions):
    """
    Ambisonic order a 3D loudspeaker layout supports, from the mean energy
    spread of VBAP at the centres of its triangles (getLayoutAmbisonicOrder).
    """
    ls = _dirs(ls_directions)
    U = _unit(ls)
    centres = _directions(np.array([U[f].sum(axis=0) for f in ConvexHull(U).simplices]))
    groups = vbap.loudspeaker_triplets(ls)
    g = vbap.gains(centres, groups, vbap.inverse_matrices(ls, groups))
    rE = np.linalg.norm(g**2 @ U, axis=1)
    spread = 2 * np.degrees(np.arccos(np.clip(rE, -1, 1)))
    order = int(np.floor(2 * 137.9 / np.mean(spread) - 1.52 + 0.5))
    return max(order, 1)


def _fzero(f, x0):
    # MATLAB fzero from a starting point: widen [x0 - dx, x0 + dx] by
    # sqrt(2) until the sign changes, then solve on that bracket.
    dx = x0 / 50 if x0 else 1 / 50
    a = b = x0
    fa = fb = f(x0)
    while (fa > 0) == (fb > 0):
        dx *= np.sqrt(2)
        a, fa = x0 - dx, f(x0 - dx)
        if (fa > 0) != (fb > 0):
            break
        b, fb = x0 + dx, f(x0 + dx)
    return brentq(f, min(a, b), max(a, b), xtol=1e-14, rtol=4 * np.finfo(float).eps)


def _energy_spread(G, U):
    E = np.sum(G**2, axis=1)
    rE = np.linalg.norm((G**2 @ U) / E[:, None], axis=1)
    return np.degrees(2 * np.arccos(np.clip(2 * rE - 1, -1, 1)))


def _csad(ls, order):
    U = _unit(ls)
    _, tdirs = t_design(4 * order)
    azi, ele = np.arange(-180, 181, 5.), np.arange(-90, 91, 5.)
    grid = np.column_stack([np.tile(azi, len(ele)), np.repeat(ele, len(azi))])
    groups = vbap.loudspeaker_triplets(ls)
    inv = vbap.inverse_matrices(ls, groups)
    max_spread = np.max(_energy_spread(vbap.energy_gains(grid, groups, inv), U))
    G = np.zeros((len(ls), len(tdirs)))
    for k, d in enumerate(tdirs):
        def excess(x):
            g = vbap.energy_gains(d, groups, inv, x, 32, 4)
            return _energy_spread(g, U)[0] - max_spread
        G[:, k] = vbap.energy_gains(d, groups, inv, _fzero(excess, max_spread), 32, 4)[0]
    Y = real_sh(order, tdirs)
    return G @ Y.T @ np.linalg.inv(Y @ Y.T)


def decoder(ls_directions, method='sad', max_re=True, order=None):
    """
    Ambisonic decoding matrix for a loudspeaker layout (ambiDecoder).

    Parameters
    ----------
    ls_directions : array_like
        (loudspeakers, 2) [azimuth, elevation] in degrees.
    method : str
        'sad' sampling (4 pi / L Y^T), 'mmd' mode matching (pseudo-inverse),
        'epad' energy preserving, 'allrad' all-round (VBAP of a 21-design,
        3D layouts) or 'csad' constant angular spread (3D layouts).
    max_re : bool
        Apply the max-rE weights (not for 'csad'). Default True.
    order : int, optional
        Decoding order; by default the order the layout supports.

    Returns
    -------
    (numpy.ndarray, int)
        D, (loudspeakers, (order + 1)^2), and the order.
    """
    ls = _dirs(ls_directions)
    if order is None:
        order = layout_order(ls)
    L = len(ls)
    Y = real_sh(order, ls)
    m = method.lower()
    if m == 'sad':
        D = 4 * np.pi / L * Y.T
    elif m == 'mmd':
        D = np.linalg.pinv(Y)
    elif m == 'epad':
        U, _, Vh = np.linalg.svd(Y)
        D = 4 * np.pi / L * Vh.T[:, :(order + 1)**2] @ U.T
    elif m == 'allrad':
        _, tdirs = t_design(21)
        groups = vbap.loudspeaker_triplets(ls)
        G = vbap.gains(tdirs, groups, vbap.inverse_matrices(ls, groups)).T
        D = 4 * np.pi / len(tdirs) * G @ real_sh(order, tdirs).T
    elif m == 'csad':
        D = _csad(ls, order)
    else:
        raise ValueError("method must be 'sad', 'mmd', 'epad', 'allrad' or 'csad'.")
    if max_re and m != 'csad':
        D = D * max_re_weights(order)
    return D, order


def _euler_zyx(alpha, beta, gamma):
    # euler2rotationMatrix(alpha, beta, gamma, 'zyx') = Rx(gamma) Ry(beta) Rz(alpha)
    c, s = np.cos, np.sin
    Rz = np.array([[c(alpha), s(alpha), 0], [-s(alpha), c(alpha), 0], [0, 0, 1]])
    Ry = np.array([[c(beta), 0, -s(beta)], [0, 1, 0], [s(beta), 0, c(beta)]])
    Rx = np.array([[1, 0, 0], [0, c(gamma), s(gamma)], [0, -s(gamma), c(gamma)]])
    return Rx @ Ry @ Rz


def sh_rotation(R, order):
    """
    Rotation matrix of real spherical harmonics up to order, from a 3 x 3
    rotation matrix R, by the recursion of Ivanic and Ruedenberg
    (getSHrotMtx, real basis).
    """
    out = np.zeros(((order + 1)**2,) * 2)
    out[0, 0] = 1
    R1 = np.array([[R[1, 1], R[1, 2], R[1, 0]],
                   [R[2, 1], R[2, 2], R[2, 0]],
                   [R[0, 1], R[0, 2], R[0, 0]]])
    if order == 0:
        return out
    out[1:4, 1:4] = R1
    prev = R1

    def P(i, l, a, b):
        if b == -l:
            return R1[i + 1, 2] * prev[a + l - 1, 0] + R1[i + 1, 0] * prev[a + l - 1, 2 * l - 2]
        if b == l:
            return R1[i + 1, 2] * prev[a + l - 1, 2 * l - 2] - R1[i + 1, 0] * prev[a + l - 1, 0]
        return R1[i + 1, 1] * prev[a + l - 1, b + l - 1]

    for l in range(2, order + 1):
        Rl = np.zeros((2 * l + 1, 2 * l + 1))
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                d = 1 if m == 0 else 0
                denom = (2 * l) * (2 * l - 1) if abs(n) == l else (l * l - n * n)
                u = np.sqrt((l * l - m * m) / denom)
                v = np.sqrt((1 + d) * (l + abs(m) - 1) * (l + abs(m)) / denom) * (1 - 2 * d) * 0.5
                w = np.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d) * (-0.5)
                if u:
                    u *= P(0, l, m, n)
                if v:
                    if m == 0:
                        v *= P(1, l, 1, n) + P(-1, l, -1, n)
                    elif m > 0:
                        dm = 1 if m == 1 else 0
                        v *= P(1, l, m - 1, n) * np.sqrt(1 + dm) - P(-1, l, -m + 1, n) * (1 - dm)
                    else:
                        dm = 1 if m == -1 else 0
                        v *= P(1, l, m + 1, n) * (1 - dm) + P(-1, l, -m - 1, n) * np.sqrt(1 + dm)
                if w:
                    w *= (P(1, l, m + 1, n) + P(-1, l, -m - 1, n) if m > 0 else
                          P(1, l, m - 1, n) - P(-1, l, -m + 1, n))
                Rl[m + l, n + l] = u + v + w
        out[l * l:(l + 1)**2, l * l:(l + 1)**2] = Rl
        prev = Rl
    return out


def rotate(hoa, yaw, pitch, roll):
    """
    Rotate a HOA sound scene, N3D, by yaw, pitch and roll in degrees
    (rotateHOA_N3D).
    """
    hoa = np.asarray(hoa, dtype=float)
    R = _euler_zyx(-np.radians(yaw), -np.radians(pitch), np.radians(roll))
    return hoa @ sh_rotation(R, _order(hoa.shape[1])).T


def rotate_bformat(bformat, yaw, pitch, roll):
    """Rotate first-order FuMa B-format by yaw, pitch and roll in degrees (rotateBformat)."""
    R = np.eye(4)
    R[1:, 1:] = _euler_zyx(-np.radians(yaw), -np.radians(pitch), np.radians(roll))
    return np.asarray(bformat, dtype=float) @ R.T
