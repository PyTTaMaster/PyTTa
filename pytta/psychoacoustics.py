"""
PyTTa psychoacoustics:
----------------------

Psychoacoustic metrics of a sound pressure signal.

Available functions:

    >>> pytta.psychoacoustics.loudness_zwicker(signal)
    >>> pytta.psychoacoustics.sharpness_din45692(signal)
    >>> pytta.psychoacoustics.roughness_daniel1997(signal)
    >>> pytta.psychoacoustics.tonality_aures1985(signal)

* loudness_zwicker: Zwicker loudness, ISO 532-1:2017, stationary and
  time-varying sounds, free or diffuse field;
* sharpness_din45692: sharpness, DIN 45692:2009, from that loudness;
* roughness_daniel1997: roughness after Daniel and Weber (1997);
* tonality_aures1985: tonality after Aures (1985), tonal components after
  Terhardt, Stoll and Seewann (1982).

Every function takes a one-channel SignalObj whose time signal is the sound
pressure in pascal, and returns a dict of floats and NumPy arrays (keys and
shapes in each docstring). For a signal stored in another unit, scale it
first (for example with SignalObj.calib_pressure).

Origin of the code
------------------

loudness_zwicker and roughness_daniel1997 are ports to Python of the MATLAB
implementations in SQAT, the Sound Quality Analysis Toolbox
(https://github.com/ggrecow/SQAT), by Gil Felix Greco, Roberto Merino-Martinez,
Alejandro Osses and Michael J. B. Lotinga. The ported files carry their own
permissive licences (the rest of SQAT is GPL-3.0 and was not used):

* Loudness_ISO532_1.m and Do_OB13_ISO532_1.m (BSD-3-Clause): MATLAB code by
  Ella Manor for AARAE (2015) after the C code of ISO 532-1:2017, Annex A.4;
  adapted and validated for SQAT by Gil Felix Greco (2023, 2025); stand-alone
  filter bank by Alejandro Osses (2023); modified by Sergio Aguirre and
  Gil Felix Greco (2026) to mirror the C reference code. The copyright notice
  and the licence are reproduced below.
* Roughness_Daniel1997.m, private/Get_Hweight_roughness.m,
  private/Get_gzi_roughness.m and the utilities Terhardt_filterbank.m,
  Terhardt_filterbank_params.m, Get_Bark.m, calculate_a0.m, get_statistics.m
  and get_exceeded_value.m (MIT): first MATLAB implementation of the Daniel
  and Weber model by J. E. Schrader at TU Eindhoven, supervised by Dik
  Hermes, then maintained by Dik Hermes; adapted for SQAT by Gil Felix Greco
  and Alejandro Osses (2014-2025), parallelised by Mike Lotinga (2025),
  restructured by Sergio Aguirre and Gil Felix Greco (2026) after the
  revised implementation of Dik Hermes (2025). The copyright notices and the
  licence are reproduced below.
* The N5 percentile of loudness_zwicker follows get_exceeded_value.m of the
  same utilities (MIT).

The port follows the MATLAB code operation by operation, so that it gives
SQAT's numbers; SQAT's resampling (MATLAB resample) is reproduced as well.

sharpness_din45692 is written from the text of DIN 45692:2009 (Formula (1)).

tonality_aures1985 is written from the papers cited below. SQAT's
Tonality_Aures1985 is GPL-3.0; it was run only as a test oracle, and of its
file only the help text (calling convention) was read. Where the papers
leave a choice open, the function docstring says what was chosen.

References
----------

ISO 532-1:2017. Acoustics - Methods for calculating loudness - Part 1:
Zwicker method.

DIN 45692:2009. Messtechnische Simulation der Hoerempfindung Schaerfe
(Measurement technique for the simulation of the auditory sensation of
sharpness).

Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of
an optimized model. Acustica, 83, 113-123.

Aures, W. (1985). Berechnungsverfahren fuer den sensorischen Wohlklang
beliebiger Schallsignale. Acustica, 59, 130-141.

Terhardt, E., Stoll, G., & Seewann, M. (1982). Algorithm for extraction of
pitch and pitch salience from complex tonal signals. The Journal of the
Acoustical Society of America, 71(3), 679-688. doi:10.1121/1.387544

Fastl, H., & Zwicker, E. (2007). Psychoacoustics: Facts and Models (3rd ed.).
Springer.

Felix Greco, G., Merino-Martinez, R., Osses, A., & Langer, S. C. (2023). SQAT:
a MATLAB-based toolbox for quantitative sound quality analysis. INTER-NOISE
and NOISE-CON Congress and Conference Proceedings, InterNoise23, Chiba,
Japan. doi:10.3397/IN_2023_1075

Licences of the ported code
---------------------------

Loudness (Loudness_ISO532_1.m, Do_OB13_ISO532_1.m), BSD-3-Clause:

    Copyright (c) <2015>, <Ella Manor>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of the <ORGANISATION> nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Roughness (Roughness_Daniel1997.m and the files it uses) and the N5
percentile (get_exceeded_value.m), MIT:

    copyright (c) 1999-2023 Dik Hermes
    copyright (c) 2025 Gil Felix Greco (get_statistics.m, get_exceeded_value.m)
    SQAT utilities Terhardt_filterbank_params.m, Get_Bark.m, calculate_a0.m:
    the SQAT authors (MIT, as stated in their headers)

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import warnings
from math import gcd

import numpy as np
import scipy.signal as ss

__all__ = ['loudness_zwicker', 'sharpness_din45692', 'roughness_daniel1997',
           'tonality_aures1985']

_FIELDS = ('free', 'diffuse')
_METHODS = ('stationary', 'time_varying')


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _mono(signal):
    """Time signal (float64, 1-D) and sampling rate of a one-channel SignalObj."""
    x = np.asarray(signal.timeSignal, dtype=float)
    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError('psychoacoustic metrics take one channel at a '
                             f'time; the signal has {x.shape[1]} channels.')
        x = x[:, 0]
    return x, signal.samplingRate


def _resample(x, fs_out, fs_in):
    """
    Resample x from fs_in to fs_out as MATLAB's resample(x, p, q) does.

    Polyphase FIR with a Kaiser window (beta 5) over 10 periods of the
    slower rate on each side, filter delay removed, output length
    ceil(len(x) * p / q). Reproduces MATLAB to rounding (2e-15 on a unit
    variance noise).
    """
    g = gcd(int(fs_out), int(fs_in))
    p, q = int(fs_out) // g, int(fs_in) // g
    pqmax = max(p, q)
    L = 2 * 10 * pqmax + 1
    m = np.arange(L) - (L - 1) / 2
    h = np.sinc(m / pqmax) * ss.windows.kaiser(L, 5.0)
    h = p * h / np.sum(h)
    Lhalf = (L - 1) / 2
    Lx = len(x)
    nz = int(np.floor(q - np.mod(Lhalf, q)))
    h = np.concatenate([np.zeros(nz), h])
    Lhalf += nz
    delay = int(np.floor(np.ceil(Lhalf) / q))
    Ly = int(np.ceil(Lx * p / q))
    nz1 = 0
    while np.ceil(((Lx - 1) * p + len(h) + nz1) / q) - delay < Ly:
        nz1 += 1
    h = np.concatenate([h, np.zeros(nz1)])
    y = ss.upfirdn(h, x, p, q)
    return y[delay:delay + Ly]


def _exceeded(values, percent):
    """Value exceeded during percent % of the time (SQAT get_exceeded_value)."""
    idx = int(np.floor((100 - percent) / 100 * values.size))
    return np.sort(values)[max(idx, 1) - 1]


def _skip_index(time, time_skip):
    """First frame of the statistics: the frame closest to time_skip."""
    return int(np.argmin(np.abs(time - time_skip)))


# ---------------------------------------------------------------------------
# Loudness, ISO 532-1:2017 (Zwicker)
# ---------------------------------------------------------------------------

# One-third-octave filter bank at 48 kHz, ISO 532-1 Tables A.1 and A.2: three
# second-order sections per band, numerators _OB_B, denominators
# [1, -2, 1] - [0, a1, a2] (offsets per band and section, below), gain per band.
_OB_B = ((1., 2., 1.), (1., 0., -1.), (1., -2., 1.))
_OB_AD = np.array([  # per band: (a1, a2) of sections 1, 2, 3
    [-6.70260e-004, 6.59453e-004, -3.75071e-004, 3.61926e-004, -3.06523e-004, 2.97634e-004],
    [-8.47258e-004, 8.30131e-004, -4.76448e-004, 4.55616e-004, -3.88773e-004, 3.74685e-004],
    [-1.07210e-003, 1.04496e-003, -6.06567e-004, 5.73553e-004, -4.94004e-004, 4.71677e-004],
    [-1.35836e-003, 1.31535e-003, -7.74327e-004, 7.22007e-004, -6.29154e-004, 5.93771e-004],
    [-1.72380e-003, 1.65564e-003, -9.91780e-004, 9.08866e-004, -8.03529e-004, 7.47455e-004],
    [-2.19188e-003, 2.08388e-003, -1.27545e-003, 1.14406e-003, -1.02976e-003, 9.40900e-004],
    [-2.79386e-003, 2.62274e-003, -1.64828e-003, 1.44006e-003, -1.32520e-003, 1.18438e-003],
    [-3.57182e-003, 3.30071e-003, -2.14252e-003, 1.81258e-003, -1.71397e-003, 1.49082e-003],
    [-4.58305e-003, 4.15355e-003, -2.80413e-003, 2.28135e-003, -2.23006e-003, 1.87646e-003],
    [-5.90655e-003, 5.22622e-003, -3.69947e-003, 2.87118e-003, -2.92205e-003, 2.36178e-003],
    [-7.65243e-003, 6.57493e-003, -4.92540e-003, 3.61318e-003, -3.86007e-003, 2.97240e-003],
    [-1.00023e-002, 8.29610e-003, -6.63788e-003, 4.55999e-003, -5.15982e-003, 3.75306e-003],
    [-1.31230e-002, 1.04220e-002, -9.02274e-003, 5.73132e-003, -6.94543e-003, 4.71734e-003],
    [-1.73693e-002, 1.30947e-002, -1.24176e-002, 7.20526e-003, -9.46002e-003, 5.93145e-003],
    [-2.31934e-002, 1.64308e-002, -1.73009e-002, 9.04761e-003, -1.30358e-002, 7.44926e-003],
    [-3.13292e-002, 2.06370e-002, -2.44342e-002, 1.13731e-002, -1.82108e-002, 9.36778e-003],
    [-4.28261e-002, 2.59325e-002, -3.49619e-002, 1.43046e-002, -2.57855e-002, 1.17912e-002],
    [-5.91733e-002, 3.25054e-002, -5.06072e-002, 1.79513e-002, -3.69401e-002, 1.48094e-002],
    [-8.26348e-002, 4.05894e-002, -7.40348e-002, 2.24476e-002, -5.34977e-002, 1.85371e-002],
    [-1.17018e-001, 5.08116e-002, -1.09516e-001, 2.81387e-002, -7.85097e-002, 2.32872e-002],
    [-1.67714e-001, 6.37872e-002, -1.63378e-001, 3.53729e-002, -1.16419e-001, 2.93723e-002],
    [-2.42528e-001, 7.98576e-002, -2.45161e-001, 4.43370e-002, -1.73972e-001, 3.70015e-002],
    [-3.53142e-001, 9.96330e-002, -3.69163e-001, 5.53535e-002, -2.61399e-001, 4.65428e-002],
    [-5.16316e-001, 1.24177e-001, -5.55473e-001, 6.89403e-002, -3.93998e-001, 5.86715e-002],
    [-7.56635e-001, 1.55023e-001, -8.34281e-001, 8.58123e-002, -5.94547e-001, 7.43960e-002],
    [-1.10165e+000, 1.91713e-001, -1.23939e+000, 1.05243e-001, -8.91666e-001, 9.40354e-002],
    [-1.58477e+000, 2.39049e-001, -1.80505e+000, 1.28794e-001, -1.32500e+000, 1.21333e-001],
    [-2.50630e+000, 1.42308e-001, -2.19464e+000, 2.76470e-001, -1.90231e+000, 1.47304e-001],
])
_OB_GAIN = np.array([
    4.30764e-011, 8.59340e-011, 1.71424e-010, 3.41944e-010,
    6.82035e-010, 1.36026e-009, 2.71261e-009, 5.40870e-009,
    1.07826e-008, 2.14910e-008, 4.28228e-008, 8.54316e-008,
    1.70009e-007, 3.38215e-007, 6.71990e-007, 1.33531e-006,
    2.65172e-006, 5.25477e-006, 1.03780e-005, 2.04870e-005,
    4.05198e-005, 7.97914e-005, 1.56511e-004, 3.04954e-004,
    5.99157e-004, 1.16544e-003, 2.27488e-003, 3.91006e-003,
])
_OB_FC = np.array([10 ** (((i - 1) - 16) / 10.) * 1000 for i in range(1, 29)])

# Table A.3: ranges (RAP) and level reductions (DLL) of the bands below 315 Hz
_RAP = np.array([45, 55, 65, 71, 80, 90, 100, 120], dtype=float)
_DLL = np.array([
    [-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
    [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
    [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
    [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
    [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
    [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
    [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
    [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0]], dtype=float)
# Table A.5 (free to diffuse field), A.6 (threshold in quiet), A.7 (critical
# band adaptation) and A.4 (a0, transmission of the middle ear)
_DDF = np.array([0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3, 2, 0, -1.4, -2, -1.9,
                 -1, 0.5, 3, 4, 4.3, 4])
_LTQ = np.array([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                dtype=float)
_DCB = np.array([-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8,
                 1.7, 1.6, 1.4, 1.2, 0.8, 0.5, 0, -0.5])
_A0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4,
                -1.5, 2, 5, 12])
# Table A.8: upper limits of the approximated critical bands (Bark); the
# 0.0001 is added as in the C code of ISO 532-1
_ZUP = [z + 0.0001 for z in (.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6,
                             12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6, 21.8,
                             22.7, 23.6, 24)]
# Table A.9: ranges of specific loudness (RNS) and steepness of the upper
# slopes (USL, one row per range, one column per critical band index 1..>7).
# Row 13, column 5 is 0.24 as in SQAT; the corrected table of ISO 532-1:2017
# gives 0.22 there. Kept at 0.24 so that the results equal SQAT's.
_RNS = [21.5, 18, 15.1, 11.5, 9, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30,
        0.22, 0.15, 0.10, 0.035, 0]
_USL = [
    [13, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
    [9, 7.5, 6, 5.1, 4.5, 4.5, 4.5, 4.5],
    [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
    [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
    [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
    [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
    [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
    [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
    [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
    [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
    [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
    [0.40, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22],
    [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
    [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]]

_I_REF = 4e-10      # (2e-5 Pa)^2
_TINY = 1e-12       # added to the intensities before the logarithm
_FS = 48000         # the filter bank is defined at this rate
_SR_LEVEL = 2000    # rate of the band levels (time-varying method)
_SR_LOUDNESS = 500  # rate of the output (time-varying method)
_NL_ITER = 24       # inner iterations of the nonlinear decay per level sample
_BARK = np.arange(1, 241) / 10


def _third_octave_bands(x):
    """Signals of the 28 one-third-octave bands (25 Hz to 12.5 kHz) at 48 kHz."""
    out = np.empty((x.size, 28))
    for n in range(28):
        y = x
        for s in range(3):
            a = (1., -2. - _OB_AD[n, 2 * s], 1. - _OB_AD[n, 2 * s + 1])
            y = ss.lfilter(_OB_B[s], a, y)
        out[:, n] = _OB_GAIN[n] * y
    return out


def _band_levels(x, time_varying, time_skip):
    """
    One-third-octave band levels (dB): one row for the stationary method
    (mean square after time_skip), one row per 0.5 ms for the time-varying
    method (squared, smoothed by three first-order low-passes, decimated).
    """
    bands = _third_octave_bands(x) ** 2
    n = x.size
    if not time_varying:
        skip = int(np.floor(time_skip * _FS))
        if skip >= n:
            raise ValueError(f'time_skip ({time_skip} s) is not shorter than '
                             f'the signal ({n / _FS} s).')
        ms = np.sum(bands[skip:n], axis=0) / (n - skip)
        return 10 * np.log10((ms + _TINY) / _I_REF)[np.newaxis, :]
    dec = _FS // _SR_LEVEL
    levels = np.empty((int(np.ceil(n / dec)), 28))
    for i in range(28):
        tau = 2 / (3 * _OB_FC[i]) if _OB_FC[i] <= 1000 else 2 / (3 * 1000.)
        a1 = np.exp(-1 / (_FS * tau))
        band = bands[:, i]
        for _ in range(3):
            band = ss.lfilter([1 - a1], [1, -a1], band)
        levels[:, i] = 10 * np.log10((band[::dec] + _TINY) / _I_REF)
    return levels


def _core_loudness(levels, diffuse):
    """Core loudness of the 21 critical bands (ISO 532-1, steps 4 to 7)."""
    nt = levels.shape[0]
    # Step 4: weighting of the 11 bands below 315 Hz (Table A.3)
    intens = np.empty((nt, 11))
    for i in range(11):
        lev = levels[:, i]
        k = np.zeros(nt, dtype=int)
        active = np.ones(nt, dtype=bool)
        for kk in range(7):
            active &= lev > _RAP[kk] - _DLL[kk, i]
            k[active] += 1
        intens[:, i] = 10 ** ((lev + _DLL[k, i]) / 10)
    # Step 5: levels of the first three critical bands
    cbi = np.stack([np.sum(intens[:, 0:6], axis=1), np.sum(intens[:, 6:9], axis=1),
                    np.sum(intens[:, 9:11], axis=1)], axis=1)
    lcb = np.zeros_like(cbi)
    pos = cbi > 0
    lcb[pos] = 10 * np.log10(cbi[pos])
    # Step 6: core loudness of each critical band
    le = levels[:, 8:28].copy()
    le[:, 0:3] = lcb
    le = le - _A0
    if diffuse:
        le = le + _DDF
    core = np.zeros((nt, 21))
    above = le > _LTQ
    mp1 = 0.0635 * 10 ** (0.025 * _LTQ)
    mp2 = ((1 - 0.25) + 0.25 * 10 ** (0.1 * (le - _DCB - _LTQ))) ** 0.25 - 1
    c = mp1 * mp2
    c[c <= 0] = 0
    core[:, :20][above] = c[above]
    # Step 7: correction of the lowest critical band
    corr = 0.4 + 0.32 * core[:, 0] ** 0.2
    corr[corr > 1] = 1
    core[:, 0] = core[:, 0] * corr
    return core


def _nonlinear_decay(core):
    """Step 8: nonlinear temporal decay of the core loudness (in place)."""
    t_short, t_long, t_var = 0.005, 0.015, 0.075
    dt = 1 / (_SR_LEVEL * _NL_ITER)
    P = (t_var + t_long) / (t_var * t_short)
    Q = 1 / (t_short * t_var)
    l1 = -P / 2 + np.sqrt(P * P / 4 - Q)
    l2 = -P / 2 - np.sqrt(P * P / 4 - Q)
    den = t_var * (l1 - l2)
    e1 = np.exp(l1 * dt)
    e2 = np.exp(l2 * dt)
    b1 = (e1 - e2) / den
    b2 = ((t_var * l2 + 1) * e1 - (t_var * l1 + 1) * e2) / den
    b3 = ((t_var * l1 + 1) * e1 - (t_var * l2 + 1) * e2) / den
    b4 = (t_var * l1 + 1) * (t_var * l2 + 1) * (e1 - e2) / den
    b5 = np.exp(-dt / t_long)
    b6 = np.exp(-dt / t_var)
    nt = core.shape[0]
    for i in range(21):
        col = core[:, i].tolist()
        uo_last = 0.0
        u2_last = 0.0
        for j in range(nt):
            ui = col[j]
            if j < nt - 1:
                delta = (col[j + 1] - ui) / _NL_ITER
                n_inner = _NL_ITER
            else:
                delta = 0.0
                n_inner = 1
            for k in range(n_inner):
                if ui < uo_last:
                    if uo_last > u2_last:
                        u2 = uo_last * b1 - u2_last * b2
                        uo = uo_last * b3 - u2_last * b4
                        if uo < ui:
                            uo = ui
                        if u2 > uo:
                            u2 = uo
                    else:
                        uo = uo_last * b5
                        if uo < ui:
                            uo = ui
                        u2 = uo
                elif abs(ui - uo_last) < 1e-5:
                    uo = ui
                    if uo > u2_last:
                        u2 = (u2_last - ui) * b6 + ui
                    else:
                        u2 = ui
                else:
                    uo = ui
                    u2 = (u2_last - ui) * b6 + ui
                uo_last = uo
                u2_last = u2
                if k == 0:
                    col[j] = uo
                ui += delta
        core[:, i] = col
    return core


def _slopes(core):
    """
    Step 9: total loudness and specific loudness (240 values, 0.1 Bark) of
    one row of core loudness, with the upper slopes of Table A.9.
    """
    N = 0.0
    z1 = 0.0
    n1 = 0.0
    iz = 0
    z = 0.1
    j = 17
    ns = [0.0] * 245  # 240 used; slack for the float accumulation of k
    for i in range(21):
        ig = min(i, 8) - 1  # column of _USL; unused for i = 0
        zup = _ZUP[i]
        cl = core[i]
        while z1 < zup:
            if n1 <= cl:
                if n1 < cl:
                    j = 0
                    while _RNS[j] >= cl and j < 17:
                        j += 1
                z2 = zup
                n2 = cl
                N += n2 * (z2 - z1)
                k = z
                while k <= z2:
                    ns[iz] = n2
                    iz += 1
                    k += 1 / 10
                z = k
            else:
                n2 = _RNS[j]
                if n2 < cl:
                    n2 = cl
                usl = _USL[j][ig]
                dz = (n1 - n2) / usl
                z2 = z1 + dz
                if z2 > zup:
                    z2 = zup
                    dz = z2 - z1
                    n2 = n1 - dz * usl
                N += dz * (n1 + n2) / 2
                k = z
                while k <= z2:
                    ns[iz] = n1 - (k - z1) * usl
                    iz += 1
                    k += 1 / 10
                z = k
            while n2 <= _RNS[j] and j < 17:
                j += 1
            z1 = z2
            n1 = n2
    return max(N, 0.0), ns[:240]


def _loudness_level(N):
    """Loudness level (phon) of a loudness N (sone), as in ISO 532-1."""
    N = np.asarray(N, dtype=float)
    LN = 40 * N ** 0.35
    big = N >= 1
    LN[big] = 40 + 10 * np.log2(N[big])
    LN[LN < 3] = 3
    return LN


def _temporal_weighting(N):
    """Step 10: two first-order low-passes (3.5 ms and 70 ms) combined."""
    dec = _FS // _SR_LEVEL
    N = np.asarray(N, dtype=float).tolist()
    out = []
    for tau in (3.5e-3, 70e-3):
        a1 = np.exp(-1 / (_SR_LEVEL * dec * tau))
        b0 = 1 - a1
        y1 = 0.0
        y = []
        for i in range(len(N)):
            x0 = N[i]
            y1 = b0 * x0 + a1 * y1
            y.append(y1)
            if i < len(N) - 1:
                xd = (N[i + 1] - x0) / dec
                for _ in range(dec - 1):
                    x0 = x0 + xd
                    y1 = b0 * x0 + a1 * y1
        out.append(np.array(y))
    return 0.47 * out[0] + 0.53 * out[1]


def _loudness_from_third_octave_levels(levels, field='free'):
    """
    Stationary loudness from the 28 one-third-octave band levels (dB SPL,
    25 Hz to 12.5 kHz), ISO 532-1, 5.2 (the method 0 of SQAT).

    Returns a dict as loudness_zwicker with method='stationary'.
    """
    levels = np.asarray(levels, dtype=float).reshape(1, 28)
    core = _core_loudness(levels, field == 'diffuse')
    N, ns = _slopes(core[0].tolist())
    return {'N': N, 'LN': float(_loudness_level([N])[0]),
            'specific': np.array(ns), 'bark': _BARK.copy()}


def loudness_zwicker(signal, field='free', method='stationary', time_skip=0.0):
    """
    Loudness after Zwicker, ISO 532-1:2017.

    Port of Loudness_ISO532_1 of SQAT (see the module docstring); results
    equal SQAT's to rounding. A signal at a rate other than 48 kHz is
    resampled to 48 kHz first, as in SQAT (MATLAB resample).

    Parameters (default), (type):
    -----------------------------

        * signal (), (SignalObj):
            One channel, sound pressure in Pa;

        * field ('free'), (str):
            'free' or 'diffuse' sound field;

        * method ('stationary'), (str):
            'stationary' (ISO 532-1, clause 5) or 'time_varying' (clause 6);

        * time_skip (0.0), (float):
            Seconds skipped at the start. Stationary method: excluded from
            the band levels (must be shorter than the signal). Time-varying
            method: excluded from Nmax and N5 only.

    Returns:
    --------

        dict. Stationary method:

            * 'N': loudness, sone (float);
            * 'LN': loudness level, phon (float);
            * 'specific': specific loudness N', sone/Bark, shape (240,);
            * 'bark': critical band rate of 'specific', 0.1 to 24 Bark in
              steps of 0.1, shape (240,).

        Time-varying method, nt output frames every 2 ms:

            * 'N': loudness N(t), sone, shape (nt,);
            * 'LN': loudness level LN(t), phon, shape (nt,);
            * 'specific': specific loudness N'(t), sone/Bark, shape (nt, 240);
            * 'time': time of each frame, s, shape (nt,);
            * 'bark': as above, shape (240,);
            * 'Nmax', 'N5': maximum loudness and loudness exceeded 5 % of
              the time from time_skip on, sone (float).

    """
    if field not in _FIELDS:
        raise ValueError(f"field must be one of {_FIELDS}, got {field!r}.")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    x, fs = _mono(signal)
    if fs != _FS:
        x = _resample(x, _FS, fs)
    time_varying = method == 'time_varying'
    levels = _band_levels(x, time_varying, time_skip)
    core = _core_loudness(levels, field == 'diffuse')
    if not time_varying:
        N, ns = _slopes(core[0].tolist())
        return {'N': N, 'LN': float(_loudness_level([N])[0]),
                'specific': np.array(ns), 'bark': _BARK.copy()}

    core = _nonlinear_decay(core)
    dec = _SR_LEVEL // _SR_LOUDNESS
    rows = core.tolist()
    N_level = np.empty(len(rows))
    spec = []
    for l, row in enumerate(rows):
        N_level[l], ns = _slopes(row)
        if l % dec == 0:
            spec.append(ns)
    N_t = _temporal_weighting(N_level)[::dec]
    time = np.arange(N_t.size) * 2e-3
    tail = N_t[_skip_index(time, time_skip):]
    return {'N': N_t, 'LN': _loudness_level(N_t), 'specific': np.array(spec),
            'time': time, 'bark': _BARK.copy(),
            'Nmax': float(np.max(tail)), 'N5': float(_exceeded(tail, 5))}


# ---------------------------------------------------------------------------
# Sharpness, DIN 45692:2009
# ---------------------------------------------------------------------------

# Normalisation constant k of Formula (1), 0.105 <= k < 0.115, meant to bring
# the reference sound (narrowband noise 920 Hz to 1080 Hz, 60 dB) to 1.00
# acum. k = 0.11 gives 1.02 to 1.03 acum for that noise synthesised with a flat
# spectrum (within the 0.05 acum tolerance of DIN 45692, 6) and keeps the 41
# test signals of Tables A.2 and A.3 within 0.45 of their tolerance. The k
# that brings the synthetic noise to exactly 1.00 (0.1069) moves the
# broadband signals 0.1 acum down and the 10.5 kHz band out of tolerance.
_K_SHARPNESS = 0.11


def _sharpness(specific):
    """
    DIN 45692 Formula (1) over the 0.1 Bark grid of ISO 532-1 (the dz of
    both integrals cancels). specific: (240,) or (nt, 240). Returns 0 where
    the loudness is zero.
    """
    z = _BARK
    g = np.where(z <= 15.8, 1.0, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)
    num = np.sum(specific * g * z, axis=-1)
    den = np.sum(specific, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = np.where(den > 0, _K_SHARPNESS * num / den, 0.0)
    return S


def sharpness_din45692(signal, field='free', method='stationary', time_skip=0.0):
    """
    Sharpness, DIN 45692:2009.

    The first moment of the specific loudness of loudness_zwicker, weighted
    by g(z) (1 up to 15.8 Bark, rising above), times k = 0.11 (Formula (1)).
    The standard is written for the free field; a diffuse field result
    should be reported as such.

    Parameters (default), (type):
    -----------------------------

        * signal (), (SignalObj):
            One channel, sound pressure in Pa;

        * field ('free'), (str):
            'free' or 'diffuse', passed to loudness_zwicker;

        * method ('stationary'), (str):
            'stationary' or 'time_varying', passed to loudness_zwicker;

        * time_skip (0.0), (float):
            Passed to loudness_zwicker; for the time-varying method it also
            starts the frames of S50.

    Returns:
    --------

        dict. Stationary method:

            * 'S': sharpness, acum (float).

        Time-varying method:

            * 'S': sharpness S(t), acum, shape (nt,), one value per 2 ms
              frame of the specific loudness (0 where the loudness is 0);
            * 'time': time of each frame, s, shape (nt,);
            * 'S50': median of S(t) from time_skip on, acum (float), the
              single value DIN 45692 (clause 6) recommends for checks.

    """
    out = loudness_zwicker(signal, field=field, method=method, time_skip=time_skip)
    S = _sharpness(out['specific'])
    if method == 'stationary':
        return {'S': float(S)}
    tail = S[_skip_index(out['time'], time_skip):]
    return {'S': S, 'time': out['time'], 'S50': float(np.median(tail))}


# ---------------------------------------------------------------------------
# Roughness, Daniel and Weber (1997)
# ---------------------------------------------------------------------------

_CHNO = 47        # half-Bark channels
_R_WINDOW = 0.2   # s, analysis window (50 % overlap)
_R_DZ = 0.5       # Bark, integration step of the specific roughness
# Calibration: Daniel and Weber (1997), Eq. 9, cal = 0.25, re-derived in SQAT
# as 0.25 / R_ref with R_ref measured on SQAT's reference signal
_R_REF = 1.006197

# Critical bands (Get_Bark.m): band, lower edge (Hz), centre (Hz), centre (Bark)
_BARK_RAW = np.array([
    [0, 0, 50, 0.5], [1, 100, 150, 1.5], [2, 200, 250, 2.5], [3, 300, 350, 3.5],
    [4, 400, 450, 4.5], [5, 510, 570, 5.5], [6, 630, 700, 6.5],
    [7, 770, 840, 7.5], [8, 920, 1000, 8.5], [9, 1080, 1170, 9.5],
    [10, 1270, 1370, 10.5], [11, 1480, 1600, 11.5], [12, 1720, 1850, 12.5],
    [13, 2000, 2150, 13.5], [14, 2320, 2500, 14.5], [15, 2700, 2900, 15.5],
    [16, 3150, 3400, 16.5], [17, 3700, 4000, 17.5], [18, 4400, 4800, 18.5],
    [19, 5300, 5800, 19.5], [20, 6400, 7000, 20.5], [21, 7700, 8500, 21.5],
    [22, 9500, 10500, 22.5], [23, 12000, 13500, 23.5],
    [24, 15500, 20000, 24.5]], dtype=float)
# Hearing threshold of the excitation (Bark, dB), Terhardt_filterbank_params.m
_HTRES = np.array([
    [0, 130], [0.01, 70], [0.17, 60], [0.8, 30], [1, 25], [1.5, 20], [2, 15],
    [3.3, 10], [4, 8.1], [5, 6.3], [6, 5], [8, 3.5], [10, 2.5], [12, 1.7],
    [13.3, 0], [15, -2.5], [16, -4], [17, -3.7], [18, -1.5], [19, 1.4],
    [20, 3.8], [21, 5], [22, 7.5], [23, 15], [24, 48], [24.5, 60], [25, 130]])
# a0 transmission factor of the outer and middle ear, free field (Bark, dB),
# Fastl and Zwicker (2007), Fig. 8.18 (calculate_a0.m, 'fastl2007')
_A0TAB = np.array([
    [0, 0], [10, 0], [12, 1.15], [13, 2.31], [14, 3.85], [15, 5.62],
    [16, 6.92], [16.5, 7.38], [17, 6.92], [18, 4.23], [18.5, 2.31], [19, 0],
    [20, -1.43], [21, -2.59], [21.5, -3.57], [22, -5.19], [22.5, -7.41],
    [23, -11.3], [23.5, -20], [24, -40], [25, -130], [26, -999]])
# g(z) of the specific roughness, table of Dik Hermes (2025), Get_gzi_roughness.m
_GR = np.array([0.15, 0.26, 0.38, 0.47, 0.54, 0.65, 0.76, 0.83, 0.90, 0.98,
                0.98, 0.90, 0.80, 0.70, 0.62, 0.54, 0.49, 0.43, 0.39, 0.35,
                0.30, 0.30, 0.30, 0.30, 0.30])
# Modulation weighting prototypes H(fmod) (Hz, weight), Get_Hweight_roughness.m
_H2 = np.array([[0, 0], [17, 0.8], [23, 0.95], [25, 0.975], [32, 1],
                [37, 0.975], [48, 0.9], [67, 0.8], [90, 0.7], [114, 0.6],
                [171, 0.4], [206, 0.3], [247, 0.2], [294, 0.1], [358, 0]])
_H5 = np.array([[0, 0], [32, 0.8], [43, 0.95], [56, 1], [69, 0.975],
                [92, 0.9], [120, 0.8], [142, 0.7], [165, 0.6], [231, 0.4],
                [277, 0.3], [331, 0.2], [397, 0.1], [502, 0]])
_H16 = np.array([[0, 0], [23.5, 0.4], [34, 0.6], [47, 0.8], [56, 0.9],
                 [63, 0.95], [79, 1], [100, 0.975], [115, 0.95], [135, 0.9],
                 [159, 0.85], [172, 0.8], [194, 0.7], [215, 0.6], [244, 0.5],
                 [290, 0.4], [348, 0.3], [415, 0.2], [500, 0.1], [645, 0]])
_H21 = np.array([[0, 0], [19, 0.4], [44, 0.8], [52.5, 0.9], [58, 0.95],
                 [75, 1], [101.5, 0.95], [114.5, 0.9], [132.5, 0.85],
                 [143.5, 0.8], [165.5, 0.7], [197.5, 0.6], [241, 0.5],
                 [290, 0.4], [348, 0.3], [415, 0.2], [500, 0.1], [645, 0]])
_H42 = np.array([[0, 0], [15, 0.4], [41, 0.8], [49, 0.9], [53, 0.965],
                 [64, 0.99], [71, 1], [88, 0.95], [94, 0.9], [106, 0.85],
                 [115, 0.8], [137, 0.7], [180, 0.6], [238, 0.5], [290, 0.4],
                 [348, 0.3], [415, 0.2], [500, 0.1], [645, 0]])


def _mround(v):
    """MATLAB round (half away from zero) for non-negative values."""
    return int(np.floor(v + 0.5))


def _excitation_params(N, fs):
    """Bins, Bark scale, thresholds and a0 of the Terhardt excitation for windows of N samples."""
    df = fs / N
    n0 = _mround(20 / df) + 1                  # 1-based bin of 20 Hz
    ntop = _mround(20e3 / df) + 1              # 1-based bin of 20 kHz
    n01 = n0 - 1
    bins = np.arange(n0, ntop + 1) - 1         # 0-based bins from 20 Hz to 20 kHz
    freqs = bins * df
    f_sorted = np.sort(np.concatenate([_BARK_RAW[:, 1], _BARK_RAW[:, 2]]))
    z_sorted = np.sort(np.concatenate([_BARK_RAW[:, 0], _BARK_RAW[:, 3]]))
    bark = np.zeros(N)                         # Bark of each 0-based bin
    bark[bins] = np.interp(freqs, f_sorted, z_sorted)
    min_exc = np.interp(bark[bins], _HTRES[:, 0], _HTRES[:, 1])
    cf = np.array([_mround(f / df) for f in _BARK_RAW[1:25, 1]]) - n01 + 1
    bf = np.array([_mround(f / df) for f in _BARK_RAW[0:25, 2]]) - n01 + 1
    min_bf = min_exc[np.sort(np.concatenate([bf, cf])) - 1]
    a0 = np.ones(N)
    a0[bins] = 10 ** (np.interp(bark[bins], _A0TAB[:, 0], _A0TAB[:, 1]) / 20)
    return {'N': N, 'n01': n01, 'bins': bins, 'freqs': freqs, 'bark': bark,
            'min_exc': min_exc, 'min_bf': min_bf, 'a0': a0}


def _roughness_params(N, fs):
    """_excitation_params plus H(fmod) and g(z) of the roughness model."""
    prm = _excitation_params(N, fs)
    hw = np.zeros((_CHNO, N))
    for proto, fmax, rows in ((_H2, 358, (2,)), (_H5, 502, (5,)),
                              (_H16, 645, (16,)), (_H21, 645, (21,)),
                              (_H42, 645, (42,))):
        last = int(np.floor((fmax / fs) * N))
        k = np.arange(3, last + 1)             # 1-based bins, DC bins 1 and 2 left out
        for r in rows:
            hw[r - 1, k - 1] = np.interp((k - 1) * fs / N, proto[:, 0], proto[:, 1])
    for lo, hi, src in ((1, 4, 2), (6, 15, 5), (17, 20, 16), (22, 41, 21),
                        (43, 47, 42)):
        for r in range(lo, hi + 1):
            if r != src:
                hw[r - 1] = hw[src - 1]
    prm['hw'] = hw
    prm['gzi'] = np.sqrt(np.interp(np.arange(1, _CHNO + 1) / 2, np.arange(25), _GR))
    return prm


def _terhardt_excitation(spec, prm):
    """
    Excitation of the 47 half-Bark channels in the time domain (Terhardt
    1979 slopes) from the calibrated spectrum of one window. Returns the
    (47, N) excitation and the number of components whose upper slope was
    clamped to zero.
    """
    N = prm['N']
    lg = np.abs(spec[prm['bins']])
    with np.errstate(divide='ignore'):
        ldb = 20 * np.log10(lg)
    which = np.nonzero(ldb > prm['min_exc'])[0]
    if which.size == 0:
        return np.zeros((_CHNO, N)), 0
    lw = ldb[which]
    zw = prm['bark'][which + prm['n01']]
    steep = -24 - (230 / prm['freqs'][which]) + (0.2 * lw)
    s2 = np.where(steep < 0, steep, 0.0)
    npos = prm['min_bf'].size
    z1 = np.floor(2 * zw).astype(int)
    z2 = np.ceil(2 * zw).astype(int)
    cols = np.arange(1, npos + 1)
    m1 = cols[np.newaxis, :] <= z1[:, np.newaxis]
    m2 = (cols[np.newaxis, :] >= z2[:, np.newaxis]) & (cols <= _CHNO)
    zk = 0.5 * cols[np.newaxis, :]
    zi = zw[:, np.newaxis]
    li = lw[:, np.newaxis]
    stemp = np.zeros((which.size, npos))
    stemp = np.where(m1, -27 * (zi - zk) + li, stemp)
    stemp = np.where(m2, s2[:, np.newaxis] * (zk - zi) + li, stemp)
    reach = (m1 | m2) & (stemp > prm['min_bf'][np.newaxis, :])
    slopes = np.zeros_like(stemp)
    slopes[reach] = 10 ** (stemp[reach] / 20)
    amp = np.zeros((which.size, _CHNO))
    lgw = lg[which]
    for i in range(1, _CHNO + 1):
        if i != 1:
            amp[:, i - 1] = slopes[:, i - 2] / lgw
        sel = z2 > i
        amp[sel, i - 1] = slopes[sel, i] / lgw[sel]
        amp[z2 == i, i - 1] = 1
        amp[z1 == i, i - 1] = 1
    etmp = np.zeros((_CHNO, N), dtype=complex)
    etmp[:, which + prm['n01']] = amp.T * spec[which + prm['n01']]
    return 2 * N * np.real(np.fft.ifft(etmp, axis=1)), int(np.sum(steep >= 0))


def _specific_roughness(ei, prm):
    """Specific roughness (asper/Bark) of the 47 channels of one window."""
    N = prm['N']
    env = np.abs(ei)
    h0 = np.mean(env, axis=1)
    hbp = 2 * np.real(np.fft.ifft(np.fft.fft(env - h0[:, np.newaxis], axis=1)
                                  * prm['hw'], axis=1))
    rms = np.sqrt(np.mean(hbp ** 2, axis=1))
    mdept = np.zeros(_CHNO)
    pos = h0 > 0
    mdept[pos] = np.minimum(rms[pos] / h0[pos], 1)
    ki = np.zeros(_CHNO - 2)
    for k in range(_CHNO - 2):
        a = hbp[k] - np.sum(hbp[k]) / N
        b = hbp[k + 2] - np.sum(hbp[k + 2]) / N
        c12 = np.dot(a, b) / (N - 1)
        den = np.sqrt((np.dot(b, b) / (N - 1)) * (np.dot(a, a) / (N - 1)))
        if den > 0:
            ki[k] = c12 / den
    g = prm['gzi']
    ri = np.zeros(_CHNO)
    ri[0] = (g[0] * mdept[0] * ki[0]) ** 2
    ri[1] = (g[1] * mdept[1] * ki[1]) ** 2
    for k in range(2, _CHNO - 2):
        ri[k] = (g[k] * mdept[k] * ki[k - 2] * ki[k]) ** 2
    ri[_CHNO - 2] = (g[_CHNO - 2] * mdept[_CHNO - 2] * ki[_CHNO - 4]) ** 2
    ri[_CHNO - 1] = (g[_CHNO - 1] * mdept[_CHNO - 1] * ki[_CHNO - 3]) ** 2
    return (0.25 / _R_REF / _R_DZ) * ri


def roughness_daniel1997(signal, time_skip=0.0):
    """
    Roughness after Daniel and Weber (1997).

    Port of Roughness_Daniel1997 of SQAT (see the module docstring); results
    equal SQAT's within the rounding of the FFT. Blackman windows of 200 ms
    with 50 % overlap; per window, the spectrum is weighted by the a0
    transmission factor, spread over 47 half-Bark channels with Terhardt's
    slopes, and the modulation depth of each channel envelope (weighted by
    H(fmod)) and the correlation of channels 1 Bark apart give the specific
    roughness. 1 asper: 1 kHz tone at 60 dB, 100 % amplitude modulated at
    70 Hz. Signals at rates other than 40960, 44100 and 48000 Hz are
    resampled to 48 kHz first, as in SQAT (MATLAB resample).

    Parameters (default), (type):
    -----------------------------

        * signal (), (SignalObj):
            One channel, sound pressure in Pa, at least 300 ms long (one
            window and one hop);

        * time_skip (0.0), (float):
            Seconds excluded from Rmean at the start.

    Returns:
    --------

        dict, nt windows:

            * 'R': roughness R(t), asper, shape (nt,);
            * 'specific': specific roughness R'(t), asper/Bark, shape (nt, 47);
            * 'time': start of each window, s, shape (nt,) (SQAT's
              convention: (first sample index, 1-based) / fs);
            * 'bark': critical band rate of 'specific', 0.5 to 23.5 Bark,
              shape (47,);
            * 'Rmean': mean of R(t) from the window closest to time_skip
              on, asper (float).

    """
    x, fs = _mono(signal)
    if fs not in (44100, 40960, 48000):
        x = _resample(x, 48000, fs)
        fs = 48000
    N = _mround(fs * _R_WINDOW)
    hop = N // 2
    nwin = int(np.floor((x.size - N) / hop))
    if nwin < 1:
        raise ValueError(f'the signal ({x.size / fs:.3f} s) is shorter than '
                         f'one window and one hop ({1.5 * _R_WINDOW} s).')
    prm = _roughness_params(N, fs)
    window = ss.windows.blackman(N, sym=False)
    l_cal = 20 * np.log10(1 / 20e-6) - 20 * np.log10(np.sqrt(2))
    amp_cal = 10 ** (l_cal / 20) * 2 / (N * np.mean(window))
    R = np.zeros(nwin)
    spec = np.zeros((nwin, _CHNO))
    time = np.zeros(nwin)
    clamped = 0
    for w in range(nwin):
        start = w * hop
        frame = np.fft.fft(x[start:start + N] * window * amp_cal) * prm['a0']
        ei, nclamp = _terhardt_excitation(frame, prm)
        clamped += nclamp > 0
        ri = _specific_roughness(ei, prm)
        spec[w] = ri
        R[w] = _R_DZ * np.sum(ri)
        time[w] = (start + 1) / fs
    if clamped:
        warnings.warn(f'Terhardt upper slope clamped to zero in {clamped} of '
                      f'{nwin} window(s): a component exceeds 120 + 1150/f dB. '
                      'The roughness there is an extrapolation; check that the '
                      'signal is in Pa.', stacklevel=2)
    return {'R': R, 'specific': spec, 'time': time,
            'bark': np.arange(1, _CHNO + 1) / 2,
            'Rmean': float(np.mean(R[_skip_index(time, time_skip):]))}


# ---------------------------------------------------------------------------
# Tonality, Aures (1985), tonal components after Terhardt et al. (1982)
# ---------------------------------------------------------------------------

_T_WINDOW = 0.16     # s, Hann frames, 50 % overlap
_T_FMIN, _T_FMAX = 20.0, 5000.0   # Hz, range of the tonal components
_T_NEIGHBOURS = (25.0, 37.5)      # Hz, Terhardt's samples i +- 2, i +- 3 at 12.5 Hz
_T_JUMP = 7.0        # dB, Terhardt et al. (1982), Eq. (2)
# c of Aures' Eq. (13), set so that a 1 kHz tone at 60 dB gives 1 t.u. (Aures
# 3.1); about 1.09 in an ideal implementation (Hastings et al. 2003)
_C_K = 1.105464


def _bark_terhardt(f):
    """Critical band rate (Bark) of f (Hz), Terhardt et al. (1982), Eq. (6)."""
    f = np.asarray(f, dtype=float) / 1000
    return 13 * np.arctan(0.76 * f) + 3.5 * np.arctan((f / 7.5) ** 2)


def _threshold_terhardt(f):
    """Threshold in quiet (dB SPL) of f (Hz), Terhardt et al. (1982), Eq. (8)."""
    f = np.asarray(f, dtype=float) / 1000
    return 3.64 * f ** -0.8 - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f ** 4


def _tonal_components(P, df):
    """
    Tonal components of one power spectrum P (Pa^2 per bin, summing to the
    power of a component over its main lobe). Returns the peak bins, the
    frequencies (Hz), levels (dB SPL) and SPL excess (dB) of every
    component found, and the half-width (bins) of the group of samples
    that represents a component.
    """
    L = 10 * np.log10(P / _I_REF + _TINY)
    j2, j3 = (_mround(h / df) for h in _T_NEIGHBOURS)
    i = np.arange(max(j3 + 1, _mround(_T_FMIN / df)),
                  min(_mround(_T_FMAX / df), P.size - 2 - j3) + 1)
    peak = (L[i - 1] < L[i]) & (L[i] >= L[i + 1])
    for j in (-j3, -j2, j2, j3):
        peak &= L[i] - L[i + j] >= _T_JUMP
    i = i[peak]
    if i.size == 0:
        return i, np.zeros(0), np.zeros(0), np.zeros(0), j3
    # frequency: Terhardt's Eq. (3), 0.46 Hz/dB at his 12.5 Hz spacing
    fc = (i + 0.46 / 12.5 * (L[i + 1] - L[i - 1])) * df
    # level: power of the main lobe of the Hann window (5 samples)
    lc = 10 * np.log10(sum(P[i + d] for d in range(-2, 3)) / _I_REF)
    # SPL excess, Terhardt et al. (1982), Eqs. (4), (5) and (7); the noise
    # skips the central samples of every component (25 Hz on each side)
    zc = _bark_terhardt(fc)
    zb = _bark_terhardt(np.arange(P.size) * df)
    noisy = np.ones(P.size, dtype=bool)
    for d in range(-j2, j2 + 1):
        noisy[i + d] = False
    lx = np.zeros(i.size)
    for u in range(i.size):
        lower = fc[u] <= fc            # f_u below f_v: lower slope of v
        s = np.where(lower, 27.0,
                     np.minimum(-24 - 230 / fc + 0.2 * lc, 0.0))
        le = lc - s * (zc - zc[u])
        others = np.arange(i.size) != u
        tones = np.sum(10 ** (le[others] / 20)) ** 2
        noise = np.sum(P[noisy & (np.abs(zb - zc[u]) <= 0.5)]) / _I_REF
        lx[u] = lc[u] - 10 * np.log10(tones + noise
                                      + 10 ** (_threshold_terhardt(fc[u]) / 10))
    return i, fc, lc, lx, j3


def _third_octave_levels(P, df):
    """Levels (dB SPL) of the 28 one-third-octave bands of a power spectrum."""
    f = np.arange(P.size) * df
    out = np.empty(28)
    for n, fm in enumerate(_OB_FC):
        band = (f >= fm * 10 ** -0.05) & (f < fm * 10 ** 0.05)
        out[n] = 10 * np.log10(np.sum(P[band]) / _I_REF + _TINY)
    return out


def tonality_aures1985(signal, field='free', time_skip=0.0):
    """
    Tonality after Aures (1985), with the tonal components of Terhardt,
    Stoll and Seewann (1982).

    Written from the two papers; SQAT's Tonality_Aures1985 is GPL-3.0 and
    served only as a test oracle. Per Hann frame of 160 ms (50 % overlap):

    * tonal components (Terhardt et al., Eqs. 1 to 3): local maxima from
      20 Hz to 5 kHz at least 7 dB above the samples 25 and 37.5 Hz away
      on both sides (Terhardt's i +- 2 and i +- 3 at his 12.5 Hz spacing),
      frequency by Eq. (3), level from the power of the main lobe;
    * SPL excess LX (Eqs. 4 to 8): level minus the masking by the other
      components (triangular excitation, slopes 27 dB/Bark and
      24 + 230/f - 0.2 L dB/Bark), by the noise of the critical band around
      the component (samples within 25 Hz of a component left out) and by
      the threshold in quiet; only components with LX > 0 count;
    * WT = sqrt(sum (w2' w3')^2) (Aures, Eqs. 9 to 11), w2' the frequency
      weighting (maximum at 700 Hz), w3' = 1 - exp(-LX / 15 dB);
    * WGr = 1 - N_Gr / N (Eq. 12): ISO 532-1 loudness (stationary, from the
      one-third-octave levels of the spectrum) of the spectrum without the
      relevant components (37.5 Hz on each side replaced by the mean of
      the next samples) over that of the whole spectrum;
    * K = c WT^0.29 WGr^0.79 (Eq. 13), c = 1.105 so that a 1 kHz tone at
      60 dB gives 1.

    The components found this way are sinusoids, so the bandwidth
    weighting w1 of Aures (Eq. 7) is 1. Aures also extracts narrowband
    noise regions (narrower than a critical band, 7 dB above the
    neighbouring bands), the only components with a bandwidth; that step
    is not described precisely enough to reproduce and is left out, so
    narrowband noise is rated by the peaks of its spectrum. Against SQAT:
    within 1e-4 for tones, 0.015 for a tone in broadband noise.

    Parameters (default), (type):
    -----------------------------

        * signal (), (SignalObj):
            One channel, sound pressure in Pa, at least 160 ms long,
            sampled above 10 kHz;

        * field ('free'), (str):
            'free' or 'diffuse', for the loudness of WGr;

        * time_skip (0.0), (float):
            Seconds excluded from Kmean at the start.

    Returns:
    --------

        dict, nt frames every 80 ms:

            * 'K': tonality K(t), t.u., shape (nt,);
            * 'WT', 'WGr': tonal and loudness weightings, shape (nt,);
            * 'time': start of each frame, s, shape (nt,);
            * 'Kmean': mean of K(t) from the frame closest to time_skip on,
              t.u. (float).

    """
    if field not in _FIELDS:
        raise ValueError(f"field must be one of {_FIELDS}, got {field!r}.")
    x, fs = _mono(signal)
    if fs <= 2 * _T_FMAX:
        raise ValueError(f'the sampling rate ({fs} Hz) must be above '
                         f'{2 * _T_FMAX:.0f} Hz.')
    N = 2 * _mround(fs * _T_WINDOW / 2)
    hop = N // 2
    if x.size < N:
        raise ValueError(f'the signal ({x.size / fs:.3f} s) is shorter than '
                         f'one frame ({_T_WINDOW} s).')
    nwin = (x.size - N) // hop + 1
    w = ss.windows.hann(N, sym=False)
    scale = 2 / (np.sum(w) ** 2 * (N * np.sum(w ** 2) / np.sum(w) ** 2))
    df = fs / N
    WT = np.zeros(nwin)
    WGr = np.zeros(nwin)
    for n in range(nwin):
        X = np.fft.rfft(x[n * hop:n * hop + N] * w)
        P = scale * np.abs(X) ** 2
        P[0] /= 2
        P[-1] /= 2
        i, fc, lc, lx, j3 = _tonal_components(P, df)
        rel = lx > 0
        w2 = 1 / np.sqrt(1 + 0.2 * (fc[rel] / 700 + 700 / fc[rel]) ** 2)
        w3 = 1 - np.exp(-lx[rel] / 15)
        WT[n] = np.sqrt(np.sum((w2 * w3) ** 2))
        noise = P.copy()
        for k in i[rel]:
            noise[k - j3:k + j3 + 1] = (P[k - j3 - 1] + P[k + j3 + 1]) / 2
        Nt = _loudness_from_third_octave_levels(_third_octave_levels(P, df), field)['N']
        Ng = _loudness_from_third_octave_levels(_third_octave_levels(noise, df), field)['N']
        WGr[n] = max(1 - Ng / Nt, 0.0) if Nt > 0 else 0.0
    K = _C_K * WT ** 0.29 * WGr ** 0.79
    time = np.arange(nwin) * hop / fs
    return {'K': K, 'WT': WT, 'WGr': WGr, 'time': time,
            'Kmean': float(np.mean(K[_skip_index(time, time_skip):]))}
