# -*- coding: utf-8 -*-

# from pytta import default
from .classes import OctFilter, ResultList
import numpy as np
# import scipy .signal as ss
import scipy.integrate as si


def __filter(signal,
             order: int = 4,
             nthOct: int = 3,
             minFreq: float = 20,
             maxFreq: float = 20000,
             refFreq: float = 1000,
             base: int = 10):
    with OctFilter(order=order,
                   nthOct=nthOct,
                   samplingRate=signal.samplingRate,
                   minFreq=minFreq,
                   maxFreq=maxFreq,
                   refFreq=refFreq,
                   base=base) as of:
        return of.filter(signal)


def __remove_init_silence(timeSignal):
    RMS = (np.mean(timeSignal**2))**0.5
    idx = np.where(np.abs(timeSignal) >= RMS)[0]
    return timeSignal[idx[0]:], idx[0]


def __remove_nonlinear(timeSignal, samplingRate):
    RMS = (np.mean(timeSignal**2))**0.5
    idx = np.where(np.abs(timeSignal[samplingRate//2:]) <= RMS)[0]
    return timeSignal[:samplingRate//2+idx[0]], samplingRate//2+idx[0]


def strip_silences(signalObj):
    """

    """
    signal, ini = __remove_init_silence(signalObj.timeSignal[:])
    signal, fin = __remove_nonlinear(signalObj.timeSignal[:],
                                     signalObj.samplingRate)
    return np.array(signal, ndmin=2)


def __cumulative_integration(timeSignal, timeVector, samplingRate):
    signal, ini = __remove_init_silence(timeSignal[:])
    signal, fin = __remove_nonlinear(signal[:], samplingRate)
    signal = signal[::-1]**2
    signal = np.array(si.cumtrapz(signal, timeVector[ini:ini+fin],
                      axis=0, initial=0)[::-1], ndmin=2).T
    return 10*np.log10(signal/np.max(np.abs(signal)))


def filtered_response(signalObj, nthOct, **kwargs):
    """

    """
    filtered = __filter(signalObj, nthOct=nthOct, **kwargs)
    signal = [strip_silences(filtered[bd])
              for bd in range(filtered.num_channels())]
    return signal


def filtered_decays(signalObj, nthOct, **kwargs):
    """

    """
    filteredObj = __filter(signalObj, nthOct=nthOct, **kwargs)
    integList = [__cumulative_integration(filteredObj.timeSignal[:, ch],
                                          filteredObj.timeVector[:],
                                          filteredObj.samplingRate)
                 for ch in range(filteredObj.num_channels())]
    return integList


def RT(decay, signalObj, nthOct, **kwargs):
    """

    """
    try:
        decay = int(decay)
        y1 = -5
        y2 = y1 - decay
    except ValueError:
        if decay in ['EDT', 'edt']:
            y1 = 0
            y2 = -10
        else:
            raise ValueError("Decay must be either 'EDT' or an integer \
                             corresponding to the amount of energy decayed to \
                             evaluate, e.g. (decay='20' | 20).")
    output = []
    for ch in range(signalObj.num_channels()):
        filtDecay = filtered_decays(signalObj[ch], nthOct, **kwargs)
        RT = []
        for bd in range(len(filtDecay)):
            x1 = np.where(filtDecay[bd] >= y1)[0][-1]
            x2 = np.where(filtDecay[bd] >= y2)[0][-1]
            RT.append(round(3*(x2/signalObj.samplingRate
                               - x1/signalObj.samplingRate), 2))
        output.append(RT)
    return output


def C(temp, signalObj, nthOct, **kwargs):
    """

    """
    try:
        temp = int(temp)*signalObj.samplingRate//1000
    except ValueError:
        raise ValueError("The temp parameter must be an integer or a string \
                         of integers, e.g. (temp='80' | 80).")
    output = []
    for ch in range(signalObj.num_channels()):
        filtResp = filtered_response(signalObj[ch], nthOct, **kwargs)
        C = []
        for bd in range(len(filtResp)):
            C.append(round(np.sum(filtResp[bd][:temp], axis=0)
                           / np.sum(filtResp[bd][temp:], axis=0)[0], 2))
        output.append(C)
    return output


def D(temp, signalObj, nthOct, **kwargs):
    """

    """
    try:
        temp = int(temp)*signalObj.samplingRate//1000
    except ValueError:
        raise ValueError("The temp parameter must be an integer or a string \
                         of integers, e.g. (temp='50' | 50).")
    output = []
    for ch in range(signalObj.num_channels()):
        filtResp = filtered_response(signalObj[ch], nthOct, **kwargs)
        D = []
        for bd in range(len(filtResp)):
            D.append(round(10*np.log10(
                        np.sum(filtResp[bd][:temp], axis=0)
                        / np.sum(filtResp[bd][:], axis=0))[0], 2))
        output.append(D)
    return output


def analyse(obj, *param, **kwargs):
    """

    """
    pass
