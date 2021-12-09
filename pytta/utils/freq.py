"""

This utility provides frequency and fractional octave frequency bands functionalities.

For syntax purposes you should start with:
	
	>>> from pytta import utils as utils

Available functions:
	
	>>> utils.freq_to_band(freq, nthOct, ref, base)
	>>> utils.fractional_octave_frequencies(nthOct = 3, freqRange = (20., 20000.), refFreq = 1000., base = 10)
	>>> utils.normalize_frequencies(freqs, samplingRate = 44100)
	>>> utils.freqs_to_center_and_edges(freqs)
	>>> utils.filter_values(freq, values, nthOct = 3)

For further information, check the docstrings for each function 
mentioned above.

Authors:
	João Vitor G. Paes joao.paes@eac.ufsm.br and
	Caroline Gaudeoso caroline.gaudeoso@eac.ufsm.br
	Rinaldi Petrolli rinaldi.petrolli@eac.ufsm.br

"""

import numpy as np
from typing import Tuple

# Cálculo de bandas de oitava a partir de 1 kHz utilizando base 10 ou 2
__nominal_frequencies = np.array([
    0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.6, 3., 0.8,
    1., 1.25, 1.6, 2., 2.5, 3.15, 4., 5., 6.3, 8., 10., 12.5, 16.,
    20., 25., 31.5, 40., 50., 63., 80., 100., 125., 160., 200., 250.,
    315., 400., 500., 630., 800., 1000., 1250., 1600., 2000., 2500.,
    3150., 4000., 5000., 6300., 8000., 10000., 12500., 16000., 20000.
])


def freq_to_band(freq: float, nthOct: int, ref: float, base: int) -> int:
    """
    Band number from frequency value.

    Parameters
    ----------
    freq : float
        The frequency value.
    nthOct : int
        How many bands per octave.
    ref : float
        Frequency of reference, or band number 0.
    base : int
        Either 10 or 2.

    Raises
    ------
    ValueError
        If base is not 10 nor 2 raises value error.

    Returns
    -------
    int
        The band number from center.

    """
    if base == 10:
        log = np.log10
        factor = 3 / 10
    elif base == 2:
        log = np.log2
        factor = 1
    else:
        raise ValueError(f"freq_to_band: unknown base value.")
    return int(np.round(log(freq / ref) * (nthOct / factor)))


def fractional_octave_frequencies(nthOct: int = 3,
                                  freqRange: Tuple[float] = (20., 20000.),
                                  refFreq: float = 1000.,
                                  base: int = 10) -> np.ndarray:
    """
    Lower, center and upper frequency values of all bands within range.

    Parameters
    ----------
    nthOct : int, optional
        bands of octave/nthOct. The default is 3.
    freqRange : Tuple[float], optional
        frequency range. These frequencies are inside the lower and higher band, respectively.
        The default is (20., 20000.).
    refFreq : float, optional
        Center frequency of center band. The default is 1000..
    base : int, optional
        Either 10 or 2. The default is 10.

    Returns
    -------
    freqs : numpy.ndarray
        Array with shape (N, 3).

    """
    if base == 10:
        factor = 3 / 10
    elif base == 2:
        factor = 1
    minFreq, maxFreq = freqRange
    minBand = freq_to_band(minFreq, nthOct, refFreq, base)
    maxBand = freq_to_band(maxFreq, nthOct, refFreq, base)
    bands = np.arange(minBand, maxBand + 1)
    freqs = np.zeros((len(bands), 3))
    nominal_frequencies = np.copy(__nominal_frequencies)
    if nthOct > 3:
        for i in range(1, int(nthOct / 3)):
            extra_nominal_frequencies = (nominal_frequencies[1:] + nominal_frequencies[:-1]) / 2
            nominal_frequencies = np.concatenate((nominal_frequencies, extra_nominal_frequencies))
            nominal_frequencies.sort(kind='mergesort')
    nthOct = 1 / nthOct
    for k, band in enumerate(bands):
        dummy = refFreq * base ** (band * nthOct * factor)
        dummy = np.sqrt((nominal_frequencies - dummy) ** 2)
        center = nominal_frequencies[np.argmin(dummy)]
        lower = center / base ** (nthOct * factor / 2)
        upper = center * base ** (nthOct * factor / 2)
        freqs[k, :] = [lower, center, upper]
    return freqs


def normalize_frequencies(freqs: np.ndarray,
                          samplingRate: int = 44100) -> np.ndarray:
    """
    Normalize frequencies for any sampling rate.

    Parameters
    ----------
    freqs : np.ndarray
        DESCRIPTION.
    samplingRate : int, optional
        DESCRIPTION. The default is 44100.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    nyq = samplingRate // 2
    return freqs / nyq


def freqs_to_center_and_edges(freqs: np.ndarray) -> Tuple[np.ndarray]:
    """
    Separate the array returned from `fractional_octave_frequencies`.

    The returned arrays corresponde to the center and edge frequencies of
    the fractional octave bands

    Parameters
    ----------
    freqs : np.ndarray
        Array returned from `fractional_octave_frequencies`.

    Returns
    -------
    center : np.ndarray
        Center frequencies of the bands.
    edges : np.ndarray
        Edge frequencies (lower and upper) of the bands.

    """
    center = freqs[:, 1].T
    edges = np.array([freqs[:, 0], freqs[:, 2]]).T
    return center, edges


def filter_values(freq, values, nthOct: int = 3):
    """
    Filters the given values into nthOct bands.

    Parameters
    ----------
    freq : ndarray
        Array containing the frequency axis.
    values : ndarray
        Array containing the magnitude values to be filtered.
    nthOct : int, optional
        Fractional octave bands that the absorption will be filtered to.

    Returns
    -------
    bands : ndarray
        An array containing the center frequencies of the available bands.
    result : ndarray
        An array containing the filtered values in the available bands.
    """
    bands = fractional_octave_frequencies(freqRange=(min(freq), max(freq)), nthOct=nthOct)
    # bands = bands[np.argwhere((bands[:, 1] >= min(freq)) & (bands[:, 1] <= max(freq)))[:, 0]]
    idx = np.array([np.argwhere((freq >= bands[a, 0]) & (freq <= bands[a, 2])) for a in
                      np.arange(0, len(bands))], dtype=object)
    result = np.array([np.sum(values[idx[a]]) / len(idx[a]) for a in np.arange(0, len(bands))], dtype=object)
    result = np.nan_to_num(result)

    return bands[:, 1], result.astype(values.dtype)

# ref: one_third_octave - Copyleft 2007-2011 luc.jaouen@matelys.com
