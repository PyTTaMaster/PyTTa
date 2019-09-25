# -*- coding: utf-8 -*-
"""
PyTTa Room Analysis:
----------------------

    This module does calculations compliant to ISO 3382-1 in order to obtain
    room acoustic paramters.

    It has an implementation of Lundeby et al. [1] algorithm to estimate the
    correction factor for the cumulative integral, as suggested by the ISO
    3382-1.

"""

import numpy as np
from numba import njit
from pytta.classes import SignalObj, OctFilter


def _filter(signal,
            order: int = 4,
            nthOct: int = 3,
            minFreq: float = 20,
            maxFreq: float = 20000,
            refFreq: float = 1000,
            base: int = 10):
    of = OctFilter(order=order,
                   nthOct=nthOct,
                   samplingRate=signal.samplingRate,
                   minFreq=minFreq,
                   maxFreq=maxFreq,
                   refFreq=refFreq,
                   base=base)
    result = of.filter(signal)
    return SignalObj(**result[0])


@njit
def T_level_profile(timeSignal, samplingRate,
                    numSamples, numChannels, nblocks=None):
    def mean_squared(x):
        return np.mean(x**2)

    if nblocks is None:
        nblocks = 100
    samples = int(numSamples//nblocks)
    profile = np.zeros((nblocks, numChannels), dtype=np.float32)
    timeStamp = np.zeros((nblocks, 1))

    for ch in range(numChannels):
        tmp = timeSignal[:, ch]
        for idx in range(nblocks):
            profile[idx, ch] = mean_squared(tmp[:samples])
            timeStamp[idx, 0] = idx*samples/samplingRate
            tmp = tmp[samples:]
    return profile, timeStamp


@njit
def T_start_sample_ISO3382(timeSignal, threshold) -> np.ndarray:
    squaredIR = timeSignal**2

    # assume the last 10% of the IR is noise, and calculate its noise level
    noiseLevel = np.mean(squaredIR[-int(len(squaredIR)//10):, :])
    # get the maximum of the signal, that is the assumed IR peak
    max_val = np.max(squaredIR)
    max_idx = np.argmax(squaredIR)
    # check if the SNR is enough to assume that the signal is an IR. If not,
    # the signal is probably not an IR, so it starts at sample 1
    idxNoShift = np.asarray([max_val < 100*noiseLevel or
                             max_idx > int(0.9*squaredIR.shape[0])])
    # less than 20dB SNR or in the "noisy" part
    if idxNoShift.any():
        print('noiseLevelCheck: The SNR too bad or \
              this is not an impulse response.')
        return

    # find the first sample that lies under the given threshold
    threshold = abs(threshold)
    startSample = 1

#    # TODO - envelope mar/pdi - check!
#    if idxNoShift:
#        print("Something wrong!")
#        return

#    % if maximum lies on the first point, then there is no point in searching
#    % for the beginning of the IR. Just return this position.
    if max_idx > 0:

        abs_dat = 10*np.log10(squaredIR[:max_idx]) \
                  - 10.*np.log10(max_val)
        lastBelowThreshold = np.where(abs_dat < threshold)[0][-1]
        if lastBelowThreshold > 0:
            startSample = lastBelowThreshold
        else:
            startSample = 1
    return startSample


@njit
def T_circular_time_shift(timeSignal, threshold=20):
    # find the first sample where inputSignal level > 20 dB or > bgNoise level
    startSample = T_start_sample_ISO3382(timeSignal, threshold)
    timeSignal = timeSignal[startSample:]
    return startSample


@njit
def T_Lundeby_correction(timeSignal, samplingRate, numSamples,
                         numChannels, timeLength):
    returnTuple = (np.float32(0), np.int32(0), np.float32(0))
    sampleShift = T_circular_time_shift(timeSignal)
    if sampleShift is None:
        return returnTuple
    winTimeLength = 0.03  # 30 ms window
    numSamples -= sampleShift  # discount shifted samples
    numParts = 5  # number of parts per 10 dB decay. N = any([3, 10])
    dBtoNoise = 10  # stop point 10 dB above first estimated background noise
    useDynRange = 20  # dynamic range

    # 1) local time average:
    blockSamples = int(winTimeLength * samplingRate)
    timeWinData, timeVecWin = T_level_profile(timeSignal, samplingRate,
                                              numSamples, numChannels,
                                              blockSamples)

    # 2) estimate noise:
    BGL = np.mean(timeWinData[-int(timeWinData.size/10):])

    # 3) regression
    startIdx = np.argmax(np.abs(timeWinData/np.max(np.abs(timeWinData))))
    stopIdx = startIdx + np.where(10*np.log10(timeWinData[startIdx+1:])
                                  >= 10*np.log10(BGL) + dBtoNoise)[0][-1]
    dynRange = 10*np.log10(timeWinData[stopIdx]) \
        - 10*np.log10(timeWinData[startIdx])
    if (stopIdx == startIdx) or (dynRange > -5)[0]:
        print("SNR too low")
        return returnTuple

    # X*c = EDC (energy decaying curve)
    X = np.ones((stopIdx-startIdx, 2), dtype=np.float32)
    X[:, 1] = timeVecWin[startIdx:stopIdx, 0]
    c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]),
                        )[0]  # rcond=None)[0]

    if (c[1] == 0)[0] or np.isnan(c).any():  # (***) c[0] e c[1] invertidos?
        print("Regression failed, T would be inf.")
        return returnTuple

    # 4) preliminary intersection
    # (***) c[0] e c[1] invertidos? CORRIGIDO
    crossingPoint = (10*np.log10(BGL) - c[1]) / c[0]
    if (crossingPoint > 2*(timeLength + sampleShift/samplingRate))[0]:
        print("Intersection point greater than signal length.")
        return returnTuple

    # 5) new local time interval length
    nBlocksInDecay = numParts * dynRange[0] / -10

    dynRangeTime = 10*np.log10(timeVecWin[stopIdx]) \
        - 10*np.log10(timeVecWin[startIdx])
    blockSamples = int(samplingRate * dynRangeTime[0] / nBlocksInDecay)

    # 6) average
    timeWinData, timeVecWin = T_level_profile(timeSignal, samplingRate,
                                              numSamples, numChannels,
                                              blockSamples)
    idxMax = np.argmax(timeWinData)

    oldCrossingPoint = 11+crossingPoint  # arbitrary higher value to enter loop
    loopCounter = 0

    while (np.abs(oldCrossingPoint - crossingPoint) > 0.0001)[0]:
        # 7) estimate background noise level (BGL)
        corrDecay = 10  # arbitrary between 5 and 10 [dB]
        # (***) 10% da resposta impulsiva inteira, não do averaged vector
        idxLast10Percent = int(len(timeWinData[-len(timeWinData)//10:]))
        # (***) conferir conta
        cmpr = (crossingPoint - corrDecay/c[1]) * samplingRate / blockSamples
        idx10dBBelowCrossPoint = np.max(np.array([1, int(cmpr[0])]))
        BGL = np.mean(timeWinData[np.min(
                np.array([idxLast10Percent,
                          idx10dBBelowCrossPoint])):])

        # 8) estimate late decay slope
        # (***) o intervalo de avaliação deveria ser estimado utilizando
        # a última inclinação calculada
        startIdx = idxMax + np.where(10*np.log10(timeWinData[idxMax:])
                                     < 10*(np.log10(BGL)
                                     + dBtoNoise
                                     + useDynRange))[0][0]
        if startIdx == idxMax:  # where returns empty
            startIdx = 0

        stopIdx = startIdx + np.where(10*np.log10(timeWinData[startIdx+1:])
                                      >= 10*np.log10(BGL)
                                      + dBtoNoise)[0][-1]
        if stopIdx == startIdx:  # where returns empty
            print("SNR too low, stopping!")
            break

        X = np.ones((stopIdx-startIdx, 2), dtype=np.float32)
        X[:, 1] = timeVecWin[startIdx:stopIdx, 0]
        c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]))[0]
        # , rcond=None)[0]
        
        # (***) c[0] e c[1] invertidos? CORRIGIDO
        if (c[0] >= 0)[0]:
            print("Regression did not work, T -> inf. Setting to 0")
            c[0] = np.inf
            break

        # 9) find crosspoint
        oldCrossingPoint = crossingPoint
        # (***) c[0] e c[1] invertidos? CORRIGIDO
        crossingPoint = (10*np.log10(BGL) - c[1]) / c[0]

        loopCounter += 1
        if loopCounter > 30:
            print("More than 30 iterations on regression, canceling.")
            break

    lateRT = -60/c[1]
    interIdx = crossingPoint * samplingRate
    return lateRT[0], np.int32(interIdx[0]), BGL


def cumulative_integration(inputSignal, **kwargs):
    timeSignal = inputSignal.timeSignal[:]
    T_circular_time_shift(timeSignal)
    hSignal = SignalObj(timeSignal,
                        inputSignal.lengthDomain,
                        inputSignal.samplingRate)
    hSignal = _filter(hSignal, **kwargs)
    for ch in range(hSignal.numChannels):
        signal = hSignal[ch]
        timeSignal = signal.timeSignal[:]
        timeVector = signal.timeVector[:]
        samplingRate = signal.samplingRate
        numSamples = signal.numSamples
        numChannels = signal.numChannels
        timeLength = signal.timeLength
        energyDecay, energyVector = energy_decay_calculation(timeSignal,
                                                             timeVector,
                                                             samplingRate,
                                                             numSamples,
                                                             numChannels,
                                                             timeLength, ch)
        yield energyDecay, energyVector


@njit
def energy_decay_calculation(timeSignal, timeVector, samplingRate, numSamples,
                             numChannels, timeLength, ch):
    lateRT, interIdx, BGL \
        = T_Lundeby_correction(timeSignal,
                               samplingRate,
                               numSamples,
                               numChannels,
                               timeLength)
    if interIdx == 0:
        interIdx = -1
    timeSignal = timeSignal[:interIdx, 0]
    timeVector = timeVector[:interIdx]
    if lateRT != 0.0:
        C = samplingRate*BGL*lateRT/(6*np.log(10))
    else:
        print("Could not estimate C factor for iteration", ch+1)
        C = 0
    sqrInv = timeSignal[::-1]**2
    energyDecayFull = np.cumsum(sqrInv)[::-1] + C
    energyDecay = energyDecayFull/energyDecayFull[0]
    return energyDecay, timeVector


def reverberation_time(decay, listEDC, samplingRate, nthOct, **kwargs):
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
    for edc, edv in listEDC:
        RT = reverb_time_regression(edc, edv, y1, y2)
        yield RT


@njit
def reverb_time_regression(energyDecay, energyVector, upperLim, lowerLim):
    first = np.where(10*np.log10(energyDecay) >= upperLim)[0][-1]
    last = np.where(10*np.log10(energyDecay) >= lowerLim)[0][-1]
    if last <= first:
        return np.nan
    # assert last > first
    X = np.ones((last-first, 2))
    X[:, 1] = energyVector[first:last]
    c = np.linalg.lstsq(X, 10*np.log10(energyDecay[first:last]))[0]
    return -60/c[1]


def clarity(temp, signalObj, nthOct, **kwargs):  # TODO
    """

    """
#    try:
#        temp = int(temp)*signalObj.samplingRate//1000
#    except ValueError:
#        raise ValueError("The temp parameter must be an integer or a string \
#                         of integers, e.g. (temp='80' | 80).")
#    output = []
#    for ch in range(signalObj.num_channels()):
#        filtResp = filtered_response(signalObj[ch], nthOct, **kwargs)
#        C = []
#        for bd in range(len(filtResp)):
#            C.append(round(np.sum(filtResp[bd][:temp], axis=0)
#                           / np.sum(filtResp[bd][temp:], axis=0)[0], 2))
#        output.append(C)
#    return output
    pass


def definition(temp, signalObj, nthOct, **kwargs):  # TODO
    """

    """
#    try:
#        temp = int(temp)*signalObj.samplingRate//1000
#    except ValueError:
#        raise ValueError("The temp parameter must be an integer or a string \
#                         of integers, e.g. (temp='50' | 50).")
#    output = []
#    for ch in range(signalObj.num_channels()):
#        filtResp = filtered_response(signalObj[ch], nthOct, **kwargs)
#        D = []
#        for bd in range(len(filtResp)):
#            D.append(round(10*np.log10(
#                        np.sum(filtResp[bd][:temp], axis=0)
#                        / np.sum(filtResp[bd][:], axis=0))[0], 2))
#        output.append(D)
#    return output
    pass


def analyse(obj, *params, **kwargs):
    """

    """
    samplingRate = obj.samplingRate
    listEDC = cumulative_integration(obj, **kwargs)
    for prm in params:
        if 'RT' in prm:
            RTdecay = prm[1]
            RT = reverberation_time(RTdecay, listEDC, samplingRate, **kwargs)
        if 'C' in prm:
            Ctemp = prm[1]
        if 'D' in prm:
            Dtemp = prm[1]
    revTimes = [rt for rt in RT]
    return
