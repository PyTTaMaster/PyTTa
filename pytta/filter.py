# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as ss
from .classes import SignalObj


""" Cálculo de bandas de oitava a partir de 1 kHz utilizando base 10"""

__nominal_frequencies = np.array([
    0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.6, 3, 0.8,
    1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10,
    12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
])


def freq_to_band(freq: float, nthOct: int, ref: float, base: int):
    return np.round(np.log10(freq/ref)*base*(nthOct/3))


def fractional_octave_frequencies(nthOct: int = 3,
                                  minFreq: float = 20.,
                                  maxFreq: float = 20000.,
                                  refFreq: float = 1000.,
                                  base: int = 10) -> np.ndarray:
    if base == 10:
        factor = 3/10
    elif base == 2:
        factor = 1/2
    minBand = freq_to_band(minFreq, nthOct, refFreq, base)
    maxBand = freq_to_band(maxFreq, nthOct, refFreq, base)
    bands = np.arange(minBand, maxBand + 1)
    freqs = np.zeros((len(bands), 3))
    nthOct = 1/nthOct
    for k, band in enumerate(bands):
        center = refFreq*base**(band*nthOct*factor)
        lower = center/base**(nthOct*factor/2)
        upper = center*base**(nthOct*factor/2)
        freqs[k, :] = [lower, center, upper]
    return freqs


def normalize_frequencies(freqs: np.ndarray,
                          samplingRate: int = 44100) -> np.ndarray:
    nyq = samplingRate//2
    return freqs/nyq


def apply_sos_filter(sos: np.ndarray, signal: np.ndarray):
    return ss.sosfilt(sos, signal, axis=0)


class OctFilter(object):
    """
    """
    def __init__(self,
                 order: int = None,
                 nthOct: int = None,
                 samplingRate: int = None,
                 minFreq: float = None,
                 maxFreq: float = None,
                 refFreq: float = None,
                 base: int = None) -> None:
        self.order = order
        self.nthOct = nthOct
        self.samplingRate = samplingRate
        self.minFreq = minFreq
        self.minBand = freq_to_band(minFreq, nthOct, refFreq, base)
        self.maxFreq = maxFreq
        self.maxBand = freq_to_band(maxFreq, nthOct, refFreq, base)
        self.refFreq = refFreq
        self.base = base
        self.sos = self.get_sos_filters()
        return

    def __freqs_to_center_and_edges(self, freqs):
        center = freqs[:, 1].T
        edges = np.array([freqs[:, 0], freqs[:, 2]]).T
        return center, edges

    def __design_sos_butter(self, bandEdges: np.ndarray,
                            order: int = 4,
                            samplingRate: int = 44100) -> np.ndarray:
        sos = np.zeros((order, 6, len(bandEdges)))
        for i, edges in enumerate(bandEdges):
            if edges[1] >= samplingRate//2:
                edges[1] = samplingRate//2 - 1
            sos[:, :, i] = ss.butter(order, [edges[0], edges[1]], 'bp',
                                     output='sos', fs=samplingRate)
        return sos

    def get_sos_filters(self) -> np.ndarray:
        freqs = fractional_octave_frequencies(self.nthOct,
                                              self.minFreq,
                                              self.maxFreq,
                                              self.refFreq,
                                              self.base)
        center, edges = self.__freqs_to_center_and_edges(freqs)
        return self.__design_sos_butter(edges, self.order, self.samplingRate)

    def filter(self, signalObj):
        if self.samplingRate != signalObj.samplingRate:
            raise ValueError("SignalObj must have same sampling\
                             rate of filter to be filtered.")
        n = self.sos.shape[2]
        output = []
        for ch in range(signalObj.num_channels()):
            filtered = np.zeros((signalObj.numSamples, n))
            for k in range(n):
                filtered[:, k] = ss.sosfilt(self.sos[:, :, k],
                                            signalObj.timeSignal[:, ch],
                                            axis=0).T
            output.append(SignalObj(filtered, 'time', self.samplingRate))
        if len(output) == 1:
            return output[0]
        else:
            return output


k1 = 12194
k2 = 20.6

k3 = 107.7
k4 = 737.9

k5 = 158.5

k6 = 0.008304630545453301
k7 = 282.7
k8 = 1160

fc = 1000


def __Ra(freq):
    Ra = (k1**2)*(freq**4) / ((freq**2 + k2**2)
                              * (((freq**2 + k3**2)
                                  * (freq**2 + k4**2))**0.5)
                              * (freq**2 + k1**2))
    return Ra


def __A(freq):
    A = round(20*np.log10(__Ra(freq)) - 20*np.log10(__Ra(fc)), 2)
    return A


def __Rb(freq):
    Rb = (k1**2)*(freq**3) / ((freq**2 + k2**2)
                              * ((freq**2 + k5**2)**0.5)
                              * (freq**2 + k1**2))
    return Rb


def __B(freq):
    B = round(20*np.log10(__Rb(freq)) - 20*np.log10(__Rb(fc)), 2)
    return B


def __Rc(freq):
    Rc = (k1**2)*(freq**2) / ((freq**2 + k2**2) * (freq**2 + k1**2))
    return Rc


def __C(freq):
    C = round(20*np.log10(__Rc(freq)) - 20*np.log10(__Rc(fc)), 2)
    return C


def __h(freq):
    h = ((1037918.48 - freq**2)**2 + 1080768.16*(freq**2))\
            / ((9837328 - freq**2)**2 + 11723776*(freq**2))
    return h


def __Rd(freq):
    Rd = (freq / k6**2) * (__h(freq)
                           / ((freq**2 + k7**2) * (freq**2 + k8**2)))**0.5
    return Rd


def __D(freq):
    D = round(20*np.log10(__Rd(freq)) - 20*np.log10(__Rd(fc)), 2)
    return D


categories = ['20', '25', '31.5', '40', '50', '63', '80', '100', '125', '160',
              '200', '250', '315', '400', '500', '630', '800', '1000', '1250',
              '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000',
              '10000', '12500', '16000', '20000']


def weighting(kind='A', nth=None, freqs=None):
    out = []
    if freqs is not None:
        for freq in freqs:
            out.append(eval('__' + kind + '(' + freq + ')'))
    elif nth is not None:
        if nth == 1:
            for val in categories[2::3]:
                out.append(eval('__' + kind + '(' + val + ')'))
        elif nth == 3:
            for val in categories:
                out.append(eval(kind+'('+val+')'))
    return np.asarray(out, ndmin=2).T
