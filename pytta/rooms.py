# -*- coding: utf-8 -*-

# from pytta import default
from .classes import OctFilter
import numpy as np
from numpy import log10, mean, max, argmax, abs, where,\
                  ones, ndarray, array, asarray
# import scipy .signal as ss
import scipy.integrate as si


def _filter(signal,
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


def T_level_profile(inputSignal):
    def RMS(x):
        return (np.mean(x**2, axis=0))**0.5

    tmpTime = inputSignal.timeSignal[:]
    perCent = 100
    samples = int(inputSignal.numSamples/perCent)
    profile = np.zeros((perCent, inputSignal.numChannels), dtype=np.float32)
    tStamp = np.zeros(perCent)

    for ch in range(inputSignal.numChannels):
        tmp = tmpTime[:, ch]
        for idx in range(perCent):
            profile[idx, ch] = RMS(tmp[:samples])
            tStamp[idx] = idx*samples/inputSignal.samplingRate
            tmp = tmp[samples:]
    return profile, tStamp


def T_start_sample_ISO3382(inputSignal, **kwargs) -> np.ndarray:
    if 'threshold' in kwargs.keys():
        threshold = kwargs['threshold']
    else:
        threshold = 20

    squaredIR = inputSignal.timeSignal**2

    # assume the last 10% of the IR is noise, and calculate its noise level
    noiseLevel = np.asarray([np.mean(
            squaredIR[-1*int(0.9*len(squaredIR)):, :], axis=0)][0])
    # get the maximum of the signal, that is the assumed IR peak
    max_val, max_idx = np.asarray([np.max(squaredIR, axis=0)][0]),\
        np.asarray([np.argmax(squaredIR, axis=0)][0])
    # check if the SNR is enough to assume that the signal is an IR. If not,
    # the signal is probably not an IR, so it starts at sample 1
    idxNoShift = np.asarray([max_val[i] < 100*noiseLevel[i]
                            or max_idx[i] > int(0.9*inputSignal[i].numSamples)
                            for i in range(len(noiseLevel))])
    # less than 20dB SNR or in the "noisy" part
    if any(idxNoShift):
        print('noiseLevelCheck: The SNR too bad or \
              this is not an impulse response.')
        return

    # find the first sample that lies under the given threshold
    threshold = abs(threshold)
    startSample = np.ones(max_val.size, dtype=np.int32)

    for idx in range(inputSignal.numChannels):
        # TODO - envelope mar/pdi - check!
        if idxNoShift[idx]:
            continue

#    % if maximum lies on the first point, then there is no point in searching
#    % for the beginning of the IR. Just return this position.
        if max_idx[idx] > 0:

            abs_dat = 10*np.log10(squaredIR[:max_idx[idx], idx]) \
                      - 10.*np.log10(max_val[idx])
            lastBelowThreshold = np.where(abs_dat < threshold)[0][-1]
            if lastBelowThreshold.size > 0:
                startSample[idx] = lastBelowThreshold
            else:
                startSample[idx] = 1
    return startSample


def T_circular_time_shift(inputSignal, **kwargs):
    # find the first sample where inputSignal level > 20 dB or > bgNoise level
    startSample = T_start_sample_ISO3382(inputSignal, **kwargs)

    # shift initial silence to the end of the array, for each (ch)annel
    for ch in range(inputSignal.numChannels):
        # dummy copy
        timeData = inputSignal.timeSignal[:, ch]
        # shift [the night away]
        timeData[:] = np.append(timeData[startSample[ch]:],
                                timeData[:startSample[ch]], axis=0)
        # update inputSignal
        inputSignal.timeSignal[:, ch] = timeData[:]
    return startSample


def T_window_nonlinear(inputSignal, profile, timeStamp):
    def dB(x):
        return 20*np.log10(x)
    cuts = np.zeros(inputSignal.numChannels, dtype=np.int32)
    for ch in range(inputSignal.numChannels):
        cuts[ch] = [int(timeStamp[idx]*inputSignal.samplingRate)
                    for idx in range(1, profile.shape[0])
                    if np.abs(dB(profile[idx, ch])
                              - dB(profile[idx-1, ch])) < 0.5][0]
        inputSignal.timeSignal[cuts[ch]:, ch] = 0
    return cuts


def T_Lundeby_correction(filteredInput) -> tuple:
    """
    Calculation of the optional C correction from ISO 3382 as proposed by
    Lundeby et al. [1]

    filteredInput is a 1-channel SignalObj of a 1/N octave filtered IR.
    This function is not intended to be directly called.

    Outputs the estimated background noise level (BGL), the estimated late
    reverberation time (lateRT), the intersection point between the estimated
    reverberation time and the BGL (interIdx) and the signal's peak to noise
    ratio (SPNR). The output comes as a tuple with the four values.
    """
    sampleShift = T_circular_time_shift(filteredInput)
    sampleShift = sampleShift[0]  # extract value from single-valued array

    # freqVec = np.ones(filteredInput.numChannels)
    winTimeLength = np.ones(filteredInput.numChannels) * 0.03  # 30 ms window

    rawTimeData = filteredInput.timeSignal**2
    nsamples, nchannels = rawTimeData.shape
    nsamples -= sampleShift  # discount shifted samples

    BGL = lateRT = interIdx = np.nan

    numParts = 5  # number of parts per 10 dB decay. N = any([3, 10])
    dBtoNoise = 10  # end of regression
    useDynRange = 20  # dynamic range for regression

    # 1) smooth:
    blockSamples = int(winTimeLength * filteredInput.samplingRate)
    nushape = (nsamples//blockSamples, blockSamples)
    timeWinData = np.mean(
                    np.reshape(
                        rawTimeData[:(nsamples//blockSamples)*blockSamples],
                        nushape), axis=0)
    timeVecWin = np.arange(0, timeWinData.shape[0]) \
        * blockSamples / filteredInput.samplingRate

    # 2) estimate noise:
    BGL = np.mean(timeWinData[-int(timeWinData.size//10):])

    # 3) regression
    startIdx = np.argmax(timeWinData)
    stopIdx, = startIdx + np.where(10*np.log10(timeWinData[startIdx+1:])
                                   > 10*np.log10(BGL) + dBtoNoise)[-1]
    dynRange = np.diff(10*np.log10(timeWinData[[startIdx, stopIdx]]))

    if (len(stopIdx) == 0) or (stopIdx == startIdx) or (dynRange > -5):
        print("SNR too low")
        return

    # X*c = EDC (energy decaying curve)
    X = np.append(np.ones((stopIdx-startIdx+1, 1)),
                  timeVecWin[startIdx:stopIdx],
                  axis=1)
    c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]))[0]

    if c[1] == 0 or any(np.isnan(c)):
        print("Regression failed, T would be inf.")
        return

    # 4) preliminary intersection
    crossingPoint = (10*np.log10(BGL) - c[0]/c[1])
    if crossingPoint > 2*(filteredInput.timeLength
                          + sampleShift/filteredInput.samplingRate):
        print("Intersection point greater than signal length.")
        return

    # 5) new local time interval length
    nBlocksDecay = np.diff(10*np.log10(timeWinData[[startIdx, stopIdx]])) \
        / (-10*numParts)
    blockSamples = np.round(filteredInput.samplingRate
                            * np.diff(timeVecWin[[startIdx, stopIdx]])
                            / nBlocksDecay)

    # 6) average
    nushape = (nsamples//blockSamples, blockSamples)
    timeWinData = np.mean(
            np.reshape(
                rawTimeData[:(nsamples//blockSamples)*blockSamples],
                nushape), axis=0)
    timeVecWin = np.arange(0, timeWinData.shape[0]) \
        * blockSamples / filteredInput.samplingRate
    idxMax = np.argmax(timeWinData)

    oldCrossingPoint = 11+crossingPoint  # arbitrary higher value to enter loop
    loopCounter = 0

    while np.abs(oldCrossingPoint - crossingPoint) > 0.01:
        # 7) estimate background noise level (BGL)
        corrDecay = 10  # arbitrary between 5 and 10 [dB]
        idxLast10Percent = int(np.round(timeWinData.shape[0]*0.9))
        idx10dBBelowCrossPoint = np.max(1,
                                        np.round((crossingPoint
                                                  - corrDecay/c[1])
                                                 * filteredInput.samplingRate
                                                 / blockSamples))
        BGL = np.mean(timeWinData[np.min(idxLast10Percent,
                                         idx10dBBelowCrossPoint):], 
                      axis=0)

        # 8) estimate late decay slope
        try:
            startIdx = idxMax - 1 + np.where(10*np.log10(timeWinData[idxMax:])
                                             < 10*(np.log10(BGL)
                                                   + dBtoNoise
                                                   + useDynRange))[0][0]
        except IndexError:  # where returns empty
            startIdx = 1

        try:
            stopIdx = np.where(10*np.log10(timeWinData[startIdx+1:])
                               < 10*np.log10(BGL) + dBtoNoise)[0][0] + startIdx
        except IndexError:  # where returns empty
            print("SNR too low, stopping!")
            break

        X = np.append(np.ones((stopIdx-startIdx+1, 1)),
                      timeVecWin[startIdx:stopIdx], axis=1)
        c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]))

        if c[1] >= 0:
            print("Regression did not work, T -> inf. Setting to 0")
            c[1] = np.inf
            break

        # 9) find crosspoint
        oldCrossingPoint = crossingPoint
        crossingPoint = (10*np.log10(BGL) - (c[0]/c[1]))

        loopCounter += 1
        if loopCounter > 30:
            print("More than 30 iterations on regression, canceling.")
            break

    lateRT = -60/c[1]
    interIdx = crossingPoint
    BGL = 10*np.log10(BGL)
    return lateRT, interIdx, BGL


def prepare_signal(inputSignal, **kwargs):
    profile, temps = T_level_profile(inputSignal)
    startSample = T_circular_time_shift(inputSignal, **kwargs)
    zeroCut = T_window_nonlinear(inputSignal, profile, temps)
    lateRT, interIdx, BGL = T_Lundeby_correction(_filter(inputSignal))
    return inputSignal, profile, temps, startSample, zeroCut, lateRT, interIdx, BGL




#def __cumulative_integration(timeSignal, timeVector, samplingRate):
#    signal, ini = __remove_init_silence(timeSignal[:])
#    signal, fin = __remove_nonlinear(signal[:], samplingRate)
#    signal = signal[::-1]**2
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


def RT(decay, signalObj, nthOct, **kwargs):
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
            RT.append(3*(x2/signalObj.samplingRate
                         - x1/signalObj.samplingRate))
        output.append(RT)
    return output


def C(temp, signalObj, nthOct, **kwargs):
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
            C.append((np.sum(filtResp[bd][:temp], axis=0)
                      / np.sum(filtResp[bd][temp:], axis=0))[0])
        output.append(C)
    return output


def D(temp, signalObj, nthOct, **kwargs):
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
            D.append(10*np.log10(np.sum(filtResp[bd][:temp], axis=0)
                                 / np.sum(filtResp[bd][:], axis=0))[0])
        output.append(D)
    return output

