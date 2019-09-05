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
from scipy import io
import pytta


def _filter(signal,
            order: int = 4,
            nthOct: int = 3,
            minFreq: float = 20,
            maxFreq: float = 20000,
            refFreq: float = 1000,
            base: int = 10):
    of = pytta.OctFilter(order=order,
                         nthOct=nthOct,
                         samplingRate=signal.samplingRate,
                         minFreq=minFreq,
                         maxFreq=maxFreq,
                         refFreq=refFreq,
                         base=base)
    result = of.filter(signal)
    return result


def T_level_profile(inputSignal, nblocks=None):
    def mean_squared(x):
        return (np.mean(x**2, axis=0))

    if nblocks is None:
        nblocks = 100
    samples = int(inputSignal.numSamples//nblocks)
    tmpSignal = inputSignal.timeSignal[:]
    numch = inputSignal.numChannels
    samplingRate = inputSignal.samplingRate

    profile = np.zeros((nblocks, numch), dtype=np.float32)
    timeStamp = np.zeros((nblocks, 1))

    for ch in range(numch):
        tmp = tmpSignal[:, ch]
        for idx in range(nblocks):
            profile[idx, ch] = mean_squared(tmp[:samples])
            timeStamp[idx, 0] = idx*samples/samplingRate
            tmp = tmp[samples:]
    return profile, timeStamp


def T_start_sample_ISO3382(inputSignal, **kwargs) -> np.ndarray:
    if 'threshold' in kwargs.keys():
        threshold = kwargs['threshold']
    else:
        threshold = 20

    squaredIR = inputSignal.timeSignal**2

    # assume the last 10% of the IR is noise, and calculate its noise level
    noiseLevel = np.asarray([np.mean(
            squaredIR[-int(len(squaredIR)//10):, :], axis=0)][0])
    # get the maximum of the signal, that is the assumed IR peak
    max_val, max_idx = np.max(squaredIR, axis=0)[0], \
        np.argmax(squaredIR, axis=0)[0]
    # check if the SNR is enough to assume that the signal is an IR. If not,
    # the signal is probably not an IR, so it starts at sample 1
    idxNoShift = np.asarray([any(max_val < 100*noiseLevel) or
                             max_idx > int(0.9*inputSignal.numSamples)])
    # less than 20dB SNR or in the "noisy" part
    if any(idxNoShift):
        print('noiseLevelCheck: The SNR too bad or \
              this is not an impulse response.')
        return

    # find the first sample that lies under the given threshold
    threshold = abs(threshold)
    startSample = 1

    # TODO - envelope mar/pdi - check!
    if idxNoShift:
        print("Something wrong!")
        return

#    % if maximum lies on the first point, then there is no point in searching
#    % for the beginning of the IR. Just return this position.
    if max_idx > 0:

        abs_dat = 10*np.log10(squaredIR[:max_idx]) \
                  - 10.*np.log10(max_val)
        lastBelowThreshold = np.where(abs_dat < threshold)[0][-1]
        if lastBelowThreshold.size > 0:
            startSample = lastBelowThreshold
        else:
            startSample = 1
    return startSample


def T_circular_time_shift(inputSignal, **kwargs):
    # find the first sample where inputSignal level > 20 dB or > bgNoise level
    startSample = T_start_sample_ISO3382(inputSignal, **kwargs)
    # shift initial silence to the end of the array, for each (ch)annel
    # dummy copy
    timeData = inputSignal.timeSignal[:]
    # shift [the night away]
    timeData = timeData[startSample:]
    # update inputSignal
    inputSignal.timeSignal = timeData[:]
    return startSample


def T_window_nonlinear(inputSignal, profile, timeStamp):
    noiseLevel = np.mean(profile[-int(profile.shape[0]/10):, :], axis=0)

    def dB(x):
        return 10*np.log10(x)
    for idx in range(profile.shape[0]-1, 0, -1):
        if dB(profile[idx]) <= dB(noiseLevel):
            cuts = int(timeStamp[idx]*inputSignal.samplingRate)
    inputSignal.timeSignal = inputSignal.timeSignal[:cuts+1]
    return cuts+1

def T_Lundeby_correction(filteredInput):
    lateRT = interIdx = BGL = np.nan
    returnTuple = (lateRT, interIdx, BGL)
    samplerate = filteredInput.samplingRate
    sampleShift = T_circular_time_shift(filteredInput)
    if sampleShift is None:
        return returnTuple
    winTimeLength = 0.03  # 30 ms window
    nsamples = filteredInput.numSamples
    nsamples -= sampleShift  # discount shifted samples
    numParts = 5  # number of parts per 10 dB decay. N = any([3, 10])
    dBtoNoise = 10  # stop point 10 dB above first estimated background noise
    useDynRange = 20  # dynamic range

    # 1) local time average:
    blockSamples = int(winTimeLength * samplerate)
    timeWinData, timeVecWin = T_level_profile(filteredInput, blockSamples)

    # 2) estimate noise:
    BGL = np.mean(timeWinData[-int(timeWinData.size/10):], axis=0)

    # 3) regression
    startIdx = np.argmax(np.abs(timeWinData/np.max(np.abs(timeWinData))))
    stopIdx = startIdx + np.where(10*np.log10(timeWinData[startIdx+1:])
                                  >= 10*np.log10(BGL) + dBtoNoise)[0][-1]

    dynRange = np.diff(10*np.log10(timeWinData[[startIdx, stopIdx]]), axis=0)
    if (stopIdx == startIdx) or (dynRange > -5):
        print("SNR too low")
        return returnTuple

    # X*c = EDC (energy decaying curve)
    X = np.append(np.ones((stopIdx-startIdx, 1)),
                  timeVecWin[startIdx:stopIdx],
                  axis=1)
    c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]),
                        rcond=None)[0]

    if c[1] == 0 or any(np.isnan(c)):
        print("Regression failed, T would be inf.")
        return returnTuple

    # 4) preliminary intersection
    crossingPoint = (10*np.log10(BGL) - c[0]) / c[1]
    if crossingPoint > 2*(filteredInput.timeLength + sampleShift/samplerate):
        print("Intersection point greater than signal length.")
        return returnTuple

    # 5) new local time interval length
    nBlocksInDecay = numParts * np.diff(
            10*np.log10(timeWinData[[startIdx, stopIdx]]), axis=0) / -10
    blockSamples = int(samplerate * np.diff(timeVecWin[[startIdx, stopIdx]],
                                            axis=0) / nBlocksInDecay)

    # 6) average
    timeWinData, timeVecWin = T_level_profile(filteredInput, blockSamples)
    idxMax = np.argmax(timeWinData)

    oldCrossingPoint = 11+crossingPoint  # arbitrary higher value to enter loop
    loopCounter = 0

    while np.abs(oldCrossingPoint - crossingPoint) > 0.0001:
        # 7) estimate background noise level (BGL)
        corrDecay = 10  # arbitrary between 5 and 10 [dB]
        idxLast10Percent = int(len(timeWinData[-len(timeWinData)//10:]))
        idx10dBBelowCrossPoint = np.max((1,
                                        int((crossingPoint - corrDecay/c[1])
                                            * samplerate / blockSamples)))
        BGL = np.mean(timeWinData[np.min((idxLast10Percent,
                                          idx10dBBelowCrossPoint)):],
                      axis=0)

        # 8) estimate late decay slope
        try:
            startIdx = idxMax + np.where(10*np.log10(timeWinData[idxMax:])
                                         < 10*(np.log10(BGL)
                                         + dBtoNoise
                                         + useDynRange))[0][0]
        except IndexError:  # where returns empty
            startIdx = 0

        try:
            stopIdx = startIdx + np.where(10*np.log10(timeWinData[startIdx+1:])
                                          >= 10*np.log10(BGL)
                                          + dBtoNoise)[0][-1]
        except IndexError:  # where returns empty
            print("SNR too low, stopping!")
            break

        X = np.append(np.ones((stopIdx-startIdx, 1)),
                      timeVecWin[startIdx:stopIdx],
                      axis=1)
        c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]),
                            rcond=None)[0]

        if c[1] >= 0:
            print("Regression did not work, T -> inf. Setting to 0")
            c[1] = np.inf
            break

        # 9) find crosspoint
        oldCrossingPoint = crossingPoint
        crossingPoint = (10*np.log10(BGL) - c[0]) / c[1]

        loopCounter += 1
        if loopCounter > 30:
            print("More than 30 iterations on regression, canceling.")
            break

    lateRT = -60/c[1]
    interIdx = int(crossingPoint * samplerate)
    BGL = BGL
    return lateRT, interIdx, BGL


# def prepare_signal(inputSignal, **kwargs):
#    #profile, temps = T_level_profile(inputSignal)
#    startSample = T_circular_time_shift(inputSignal, **kwargs)
#    #zeroCut = T_window_nonlinear(inputSignal, profile, temps)
#    return (profile, temps, startSample)  # , zeroCut)


def cumulative_integration(inputSignal, **kwargs):
    hSignal = pytta.SignalObj(inputSignal.timeSignal,
                              inputSignal.lengthDomain,
                              inputSignal.samplingRate)
#    profile, profTemps, shiftSample, zeroCut \
#    profile, profTemps, shiftSample \
#    = prepare_signal(hSignal, **kwargs)
    _ = T_circular_time_shift(hSignal)
    hSignal = _filter(hSignal, **kwargs)
#   out = []
    for ch in range(hSignal.numChannels):
        signal = hSignal[ch]
#        timeVector = signal.timeVector[:]
        lateRT, interIdx, BGL \
            = T_Lundeby_correction(signal)
        if interIdx is np.nan:
            interIdx = -1
        signal.timeSignal = np.array(signal.timeSignal[:interIdx, 0],
                                     ndmin=2).T
        if lateRT is not np.nan:
            C = signal.samplingRate*BGL*lateRT/(6*np.log(10))
        else:
            print("Could not estimate C factor for iteration", ch+1)
            C = 0
        energyDecay = signal.timeSignal[::-1]**2
        energyDecay = np.cumsum(energyDecay, axis=0)[::-1] + C
        energyDecay *= 1/energyDecay[0]
        yield energyDecay
#        out.append(energyDecay)
#    return out

# ---*-----
#    signal = np.array(si.cumtrapz(signal, timeVector[ini:ini+fin],
#                      axis=0, initial=0)[::-1], ndmin=2).T
#    return 10*np.log10(signal/np.max(np.abs(signal)))
#
#
#def filtered_response(signalObj, nthOct, **kwargs):
#    filtered = __filter(signalObj, nthOct=nthOct, **kwargs)
#    signal = [strip_silences(filtered[bd])
#              for bd in range(filtered.num_channels())]
#    return signal
#
#
#def filtered_decays(signalObj, nthOct, **kwargs):
#    filteredObj = __filter(signalObj, nthOct=nthOct, **kwargs)
#    integList = [__cumulative_integration(filteredObj.timeSignal[:, ch],
#                                          filteredObj.timeVector[:],
#                                          filteredObj.samplingRate)
#                 for ch in range(filteredObj.num_channels())]
#    return integList


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
#    output = []
    for edc in listEDC:
        x1 = np.argmin(10*np.log10(edc) >= y1)
        x2 = np.argmin(10*np.log10(edc) >= y2)
        RT = round(3*(x2/samplingRate - x1/samplingRate), 2)
        yield RT
#        output.append(RT)
#    return output


def clarity(temp, signalObj, nthOct, **kwargs):
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


def definition(temp, signalObj, nthOct, **kwargs):
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
