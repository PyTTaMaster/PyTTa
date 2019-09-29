"""
__module Frequency Utilities:
----------------------

    Provides frequency and fractional octave frequency bands functionalities

    Functions:

        :param:freq_to_band

"""

import numpy as np

""" Cálculo de bandas de oitava a partir de 1 kHz utilizando base 10 ou 2"""
__nominal_frequencies = np.array([
    0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.6, 3, 0.8,
    1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10,
    12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
])


def freq_to_band(freq: float, nthOct: int, ref: float, base: int) -> np.ndarray:
    log = lambda x: np.log(x)/np.log(base)
    factor = np.log2(base)
    return np.round(log(freq / ref) * (nthOct / factor))


def fractional_octave_frequencies(nthOct: int = 3,
                                  minFreq: float = 20.,
                                  maxFreq: float = 20000.,
                                  refFreq: float = 1000.,
                                  base: int = 10) -> np.ndarray:
    if base == 10:
        factor = 3 / 10
    elif base == 2:
        factor = 1
    minBand = freq_to_band(minFreq, nthOct, refFreq, base)
    maxBand = freq_to_band(maxFreq, nthOct, refFreq, base)
    bands = np.arange(minBand, maxBand + 1)
    freqs = np.zeros((len(bands), 3))
    nthOct = 1 / nthOct
    for k, band in enumerate(bands):
        dummy = refFreq * base ** (band * nthOct * factor)
        dummy = np.sqrt((__nominal_frequencies - dummy) ** 2)
        center = __nominal_frequencies[np.argmin(dummy)]
        lower = center / base ** (nthOct * factor / 2)
        upper = center * base ** (nthOct * factor / 2)
        freqs[k, :] = [lower, center, upper]
    return freqs


def normalize_frequencies(freqs: np.ndarray,
                          samplingRate: int = 44100) -> np.ndarray:
    nyq = samplingRate // 2
    return freqs / nyq


def freqs_to_center_and_edges(freqs):
    center = freqs[:, 1].T
    edges = np.array([freqs[:, 0], freqs[:, 2]]).T
    return center, edges
