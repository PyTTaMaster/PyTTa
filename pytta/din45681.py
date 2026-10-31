"""
PyTTa DIN 45681 tonal audibility:
---------------------------------

Detection of tonal components in noise and their audibility, DIN 45681:2002,
as computed by ita_tonality of the ITA-Toolbox.

Available functions:

    >>> pytta.din45681.tonal_audibility(signal)

Origin of the code
------------------

tonal_audibility is a port to Python of ita_tonality.m of the ITA-Toolbox
(applications/Psychoacoustics), by Daniel Cragg, Institute of Technical
Acoustics, RWTH Aachen University (2010). The method is the same; three
points differ from the MATLAB code and are stated in the docstring: the
frequency of each FFT line (ita_tonality labels line k as (k + 1) df, so
every frequency is 2 Hz high), the absolute scale of the levels (6 dB low
there, for tones and noise alike, so DeltaL does not change), and the ends of
the search for a tone region, which stop at the analysed range. The licence
of the ITA-Toolbox is reproduced below.

This product includes software developed by the Institute of Technical
Acoustics (RWTH Aachen University).

References
----------

DIN 45681:2002. Akustik - Bestimmung der Tonhaltigkeit von Geraeuschen und
Ermittlung eines Tonzuschlages fuer die Beurteilung von Geraeuschimmissionen.

Licence of the ported code (ITA-Toolbox, license.txt)
-----------------------------------------------------

    Copyright (c) 2011
    Institute of Technical Acoustics (RWTH Aachen University)
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in
       the documentation and/or other materials provided with the
       distribution.
    3. All advertising materials mentioning features or use of this
       software must display the following acknowledgment:

       This product includes software developed by the
       Institute of Technical Acoustics (RWTH Aachen University).

    4. Neither the name of the Institute of Technical Acoustics (RWTH
       Aachen University), nor the names of its contributors may be used
       to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE INSTITUTE OF TECHNICAL ACOUSTICS
    (RWTH AACHEN UNIVERSITY) ''AS IS'' AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL INSTITUTE OF TECHNICAL ACOUSTICS (RWTH AACHEN
    UNIVERSITY) BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
    OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
"""

import numpy as np

from pytta.classes import SignalObj

_P0 = 2e-5
_DF = 2.0           # line spacing, Hz
_FMIN, _FMAX = 100., 8192.
_HANN = 10 * np.log10(1.5)   # equivalent noise bandwidth of the Hann window


def _a_weighting(f):
    # A weighting as written in ita_tonality (12200 Hz pole, 165.5 dB).
    return 165.5 + 20 * np.log10(f**4 / (f**2 + 20.6**2) / (f**2 + 12200**2)
                                 / np.sqrt(f**2 + 107.7**2)
                                 / np.sqrt(f**2 + 737.9**2))


def tonal_audibility(signal):
    """
    Tonal components of a noise and their audibility, DIN 45681:2002.

    The A-weighted power spectrum is averaged over Hann windowed blocks of
    fs / 2 samples (lines of 2 Hz), with a hop of 10000 samples at 44.1 kHz
    (the same fraction of the block at other rates). For each line from
    100 Hz to 8192 Hz the masking noise level LS of its critical band,
    dfc = 25 + 75 (1 + 1.4 (f/1 kHz)^2)^0.69 Hz, is the energy mean of the
    lines not more than 6 dB above LS, iterated until it settles. A
    contiguous region of lines 6 dB or more above LS is a tone candidate at
    its highest line; its level LT adds the neighbour lines within 10 dB of
    the peak. With LG = LS + 10 lg(dfc / df) and the audibility threshold
    av = -2 - lg(1 + (f/502)^2.5), the audibility is

        DeltaL = LT - LG - av.

    A candidate is no tone, and gets DeltaL = 0, when it is wider than
    18 (1 + 0.001 f) Hz or when one of its flanks is less steep than 24.
    Tones in one critical band are summed into DeltaL_band.

    Differences from ita_tonality: frequencies are those of the FFT lines
    (ita_tonality reads each one 2 Hz high); levels are absolute (6 dB low
    there for tones and noise, which leaves DeltaL as it is); the search for
    a region stops at 100 Hz and 8192 Hz. The tone adjustment K_T of the
    standard is not computed.

    Parameters
    ----------
    signal : SignalObj
        One channel of sound pressure in Pa, at least one block (0.5 s).

    Returns
    -------
    dict
        'deltaL': the largest of DeltaL and DeltaL_band (dB), 0 without tones;
        'frequency': the frequency of that tone (Hz), None without tones;
        'tones': one dict per tone with 'frequency', 'bandwidth' (dfc),
        'LS', 'LT', 'LG', 'av', 'deltaL' and 'deltaL_band' (dB, Hz).

    >>> out = pytta.din45681.tonal_audibility(recording)
    """
    if not isinstance(signal, SignalObj):
        raise TypeError('signal must be a SignalObj.')
    if signal.numChannels != 1:
        raise ValueError('signal must have one channel.')
    fs = signal.samplingRate
    N = int(round(fs / _DF))
    df = fs / N
    hop = int(round(N * 10000 / 22050))
    x = np.asarray(signal.timeSignal[:, 0], dtype=float)
    starts = np.arange(0, x.size - N + 1, hop)
    if starts.size == 0:
        raise ValueError('signal is shorter than one block of 0.5 s.')
    w = np.hanning(N + 2)[1:-1]            # MATLAB hanning(N)
    blocks = np.stack([x[s:s + N] * w for s in starts])
    X = np.abs(np.fft.rfft(blocks, axis=1))[:, :N // 2]
    power = np.mean(2 * X**2 / np.sum(w)**2, axis=0) / _P0**2
    freq = np.arange(N // 2) * df
    with np.errstate(divide='ignore'):
        level = 10 * np.log10(power)
        level[1:] += _a_weighting(freq[1:])
    level[0] = -np.inf

    first, last = int(round(_FMIN / df)), int(round(_FMAX / df))
    idx = np.arange(first, last + 1)
    dfc = 25 + 75 * (1 + 1.4 * (freq / 1000)**2)**0.69
    lo = np.clip(np.searchsorted(freq, freq - dfc / 2), 1, last)
    hi = np.clip(np.searchsorted(freq, freq + dfc / 2) - 1, None, last)

    LS = np.full(freq.size, np.inf)
    for i in idx:                           # masking noise level
        band = level[lo[i]:hi[i] + 1]
        ls, previous, lines = 1000., 0., 1000
        while lines > 9 and previous != ls:
            keep = band <= ls + 6
            previous, lines = ls, int(np.sum(keep))
            if lines:
                ls = 10 * np.log10(np.mean(10**(band[keep] / 10))) - _HANN
        LS[i] = previous if lines < 10 else ls

    above = level >= LS + 6
    deltaL, LT, LG, av = {}, {}, {}, {}
    i = first
    while i <= last:                        # tone candidates
        if not above[i]:
            i += 1
            continue
        end = i
        while end + 1 <= last and above[end + 1]:
            end += 1
        k = i + int(np.argmax(level[i:end + 1]))
        ref = LS[k] + 6
        tone, nlines = 10**(level[k] / 10), 1
        j = k - 1
        while level[j] >= ref and level[j] >= level[k] - 10:
            tone, nlines, j = tone + 10**(level[j] / 10), nlines + 1, j - 1
        lower = ((level[k] - level[j]) * freq[k] / np.sqrt(2) / (freq[k] - freq[j])
                 if j < k - 1 else 100)
        j = k + 1
        while level[j] >= ref and level[j] >= level[k] - 10:
            tone, nlines, j = tone + 10**(level[j] / 10), nlines + 1, j + 1
        upper = ((level[k] - level[j]) * freq[k] / np.sqrt(2) / (freq[j] - freq[k])
                 if j > k + 1 else 100)
        LT[k] = 10 * np.log10(tone) - (_HANN if nlines > 1 else 0)
        LG[k] = LS[k] + 10 * np.log10(dfc[k] / df)
        av[k] = -2 - np.log10(1 + (freq[k] / 502)**2.5)
        deltaL[k] = LT[k] - LG[k] - av[k]
        if (nlines * df > 18 * (1 + 0.001 * freq[k]) or lower < 24
                or upper < 24):
            deltaL[k] = 0.
        i = end + 1

    tones = []
    for k in sorted(k for k in deltaL if deltaL[k] > 0):
        others = [m for m in deltaL if m != k and deltaL[m] > 0
                  and lo[k] <= m <= hi[k]]
        band = (10 * np.log10(sum(10**(LT[m] / 10) for m in [k] + others))
                - LG[k] - av[k]) if others else deltaL[k]
        tones.append(dict(frequency=freq[k], bandwidth=dfc[k], LS=LS[k],
                          LT=LT[k], LG=LG[k], av=av[k], deltaL=deltaL[k],
                          deltaL_band=band))
    if not tones:
        return dict(deltaL=0, frequency=None, tones=[])
    best = max(tones, key=lambda t: max(t['deltaL'], t['deltaL_band']))
    return dict(deltaL=max(best['deltaL'], best['deltaL_band']),
                frequency=best['frequency'], tones=tones)
