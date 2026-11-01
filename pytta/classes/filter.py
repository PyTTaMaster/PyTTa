# -*- coding: utf-8 -*-

import numpy as np
from copy import copy
from scipy import signal as ss
from pytta.classes import SignalObj
from pytta.classes._base import ChannelsList
from pytta.utils import fractional_octave_frequencies, freq_to_band, \
                        normalize_frequencies, freqs_to_center_and_edges


class FilterBase(object):
    """Base class for filters."""

    def __init__(self, order: int, samplingrate: int):
        self.samplingRate = samplingrate
        self.order = order
        return

    def __call__(self, sigobj: SignalObj):
        """
        Filter the signal object.

        For each channel inside the input signalObj, will be generated a new SignalObj with the channel filtered signal.

        Args:
            sigobj: SignalObj

        Return:
            output: List
                A list containing one SignalObj with the filtered data for each channel in the original signalObj.

        """
        return self.filter(sigobj)



class OctFilter(FilterBase):
    """
    Octave filter.
    """
    def __init__(self,
                 order: int = None,
                 nthOct: int = None,
                 samplingRate: int = None,
                 minFreq: float = None,
                 maxFreq: float = None,
                 refFreq: float = None,
                 base: int = None) -> None:
        """

        Parameters
        ----------
        order : int, optional
            DESCRIPTION. The default is None.
        nthOct : int, optional
            DESCRIPTION. The default is None.
        samplingRate : int, optional
            DESCRIPTION. The default is None.
        minFreq : float, optional
            DESCRIPTION. The default is None.
        maxFreq : float, optional
            DESCRIPTION. The default is None.
        refFreq : float, optional
            DESCRIPTION. The default is None.
        base : int, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None
            DESCRIPTION.

        """
        FilterBase.__init__(self, order, samplingRate)
        self.nthOct = nthOct
        self.minFreq = minFreq
        self.minBand = freq_to_band(minFreq, nthOct, refFreq, base)
        self.maxFreq = maxFreq
        self.maxBand = freq_to_band(maxFreq, nthOct, refFreq, base)
        self.refFreq = refFreq
        self.base = base
        self.sos = self.get_sos_filters()
        return

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if traceback is None:
            return
        else:
            raise value

    def __design_sos_butter(self,
                            bandEdges: np.ndarray,
                            order: int = 4,
                            samplingRate: int = 44100) -> np.ndarray:
        sos = np.zeros((order, 6, len(bandEdges)))
        for i, edges in enumerate(bandEdges):
            if edges[1] >= samplingRate//2:
                edges[1] = samplingRate//2 - 1
            sos[:, :, i] = ss.butter(N=order, Wn=np.array([edges[0],
                                                          edges[1]]),
                                     btype='bp', output='sos', fs=samplingRate)
        return sos

    def get_sos_filters(self) -> np.ndarray:
        freqs = fractional_octave_frequencies(self.nthOct,
                                              (self.minFreq,
                                               self.maxFreq),
                                              self.refFreq,
                                              self.base)
        self.center, edges = freqs_to_center_and_edges(freqs)
        return self.__design_sos_butter(edges, self.order, self.samplingRate)

    # def filter(self, sigobj):
    #     print(":WARNING: `OctFilter.filter` method will soon be deprecated.")
    #     return self._filter(sigobj)

    def filter(self, sigobj):
        """
        Filter the signal object.

        For each channel inside the input signalObj, will be generated a new
        SignalObj with the channel filtered signal.

        Args:
            sigobj: SignalObj

        Return:
            output: List
                A list containing one SignalObj with the filtered data for each
                channel in the original signalObj.

        """
        if self.samplingRate != sigobj.samplingRate:
            raise ValueError("SignalObj must have same sampling rate of filter to be filtered.")
        n = self.sos.shape[2]
        output = []
        chl = []
        for ch in range(sigobj.numChannels):
            sobj = sigobj[ch]
            filtered = np.zeros((sobj.numSamples, n))
            for k in range(n):
                cContigousArray = sobj.timeSignal[:].copy(order='C')
                filtered[:, k] = ss.sosfilt(self.sos[:, :, k].copy(order='C'),
                                            cContigousArray,
                                            axis=0).T
                chl.append(copy(sobj.channels[sobj.channels.mapping[0]]))
                chl[-1].num = k+1
                chl[-1].name = f'Band {k+1}'
                chl[-1].code = f'B{k+1}'
            signalDict = {'signalArray': filtered,
                          'domain': 'time',
                          'samplingRate': self.samplingRate,
                          'freqMin': sigobj.freqMin,
                          'freqMax': sigobj.freqMax,
                          }
            out = SignalObj(**signalDict)
            out.channels = ChannelsList(chl)
            # out.timeSignal = out.timeSignal * out.channels.CFlist()
            output.append(out)
            chl.clear()
        return output



class AntiAliasingFilter(object):
    def __init__(self, order, band, samplingRate):
        self.order = order
        self.band = band
        self.samplingRate = samplingRate
        self.__design()
        return

    def __design(self):
        self.sos = np.zeros((self.order, 6))
        if self.band[1] > 22050:
            self.band[1] = 22050
        elif self.band[0] < 12.5:
            self.band[0] = 12.5
        self.sos[:, :] = ss.butter(N=self.order, Wn=np.array(self.band), btype='bp', output='sos', fs=self.samplingRate)
        return

    def filter(self, signalObj):
        if self.samplingRate != signalObj.samplingRate:
            raise ValueError("SignalObj must have same sampling\
                             rate of filter to be filtered.")
        output = []
        for ch in range(signalObj.numChannels):
            filtered = np.zeros((signalObj.numSamples))
            filtered[:] = ss.sosfilt(self.sos[:, :],
                                     signalObj.timeSignal[:, ch],
                                     axis=0).T
            signalDict = {'signalArray': filtered,
                          'domain': 'time',
                          'samplingRate': self.samplingRate,
                          'freqMin': self.band[0],
                          'freqMax': self.band[1]}
            output.append(SignalObj(**signalDict))
        else:
            return output


#class SPLWeight(object):

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


_categories = ['20', '25', '31.5', '40', '50', '63', '80', '100', '125', '160',
               '200', '250', '315', '400', '500', '630', '800', '1000', '1250',
               '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000',
               '10000', '12500', '16000', '20000']


def weighting(kind='A', nth=None, freqs=None):
    """
    Frequency weighting in dB, from the analytical expressions of IEC
    61672-1:2013, Annex E ('A' and 'C'; 'B' and 'D' of the former
    standards), normalized to 0 dB at 1 kHz.

    Parameters
    ----------
    kind : str, optional
        'A', 'B', 'C' or 'D'. The default is 'A'.
    nth : int, optional
        1 or 3: the weighting of the octave bands from 31.5 Hz to 16 kHz or
        of the one-third octave bands from 20 Hz to 20 kHz, at the exact
        frequencies 1000 * 10**(0.1 (n - 30)) at which IEC 61672-1, Table 3,
        gives it for the nominal ones.
    freqs : array_like, optional
        Frequencies in Hz, used instead of nth.

    Returns
    -------
    np.ndarray
        The weighting in dB, one row per frequency, rounded to 0.01 dB.

    """
    curves = {'A': __A, 'B': __B, 'C': __C, 'D': __D}
    if kind not in curves:
        raise ValueError("kind must be 'A', 'B', 'C' or 'D'.")
    if freqs is None:
        if nth not in (1, 3):
            raise ValueError('Give freqs, or nth = 1 or 3.')
        n = np.arange(13, 44)                    # 20 Hz to 20 kHz
        freqs = 1000 * 10 ** (0.1 * (n - 30))
        if nth == 1:
            freqs = freqs[2::3]                  # 31.5 Hz to 16 kHz
    out = [curves[kind](float(f)) for f in np.atleast_1d(freqs)]
    return np.array(out, ndmin=2).T
