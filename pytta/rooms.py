# -*- coding: utf-8 -*-

"""
This module does calculations compliant to ISO 3382-1 to obtain room acoustic parameters.

It has an implementation of Lundeby et al. [1] algorithm
to estimate the correction factor for the cumulative integral, as suggested
by the ISO 3382-1.

Use this module through the function 'analyse', which receives an one channel
SignalObj or ImpulsiveResponse and calculate the room acoustic parameters
especified in the positional input arguments. For more information check
pytta.rooms.analyse's documentation.

Please, use and test the RoomParameters class, which provides several
energy parameters for monaural room impulse response.

Available functions:

    >>> pytta.rooms.crop_IR(SignalObj | ImpulsiveResponse, ...)
    >>> pytta.rooms.Analyse(SignalObj, ...)
    >>> pytta.rooms.strength_factor(...)
    >>> pytta.rooms.G_Lpe
    >>> pytta.rooms.G_Lps
    >>> pytta.rooms.RoomParameters(SignalObj, ...)

Authors:
    João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
    Matheus Lazarin, matheus.lazarin@eac.ufsm.br
    Rinaldi Petrolli, rinaldi.petrolli@eac.ufsm.br

"""

import numpy as np
import matplotlib.pyplot as plt
# from numba import njit
from pytta import SignalObj, OctFilter, Analysis, ImpulsiveResponse
from pytta.utils import fractional_octave_frequencies as FOF, freq_to_band
import traceback
import copy as cp


class RoomParameters(Analysis):
    """Room monaural parameters."""

    def __init__(self, ir: SignalObj, nthOct: int = 1,
                 minFreq: float = 2e1, maxFreq: float = 2e4, *args,
                 bypassLundeby: bool = False, suppressWarnings: bool = True,
                 ircut: float = None, **kwargs):
        """
        Room acoustical parameters for quality analysis.

        Provides interface to estimate several room parameters based on
        the energy distribution of the impulse response.

        Parameters
        ----------
        ir : SignalObj
            Monaural room impulse response signal.
        nthOct : int, optional
            Number of bands per octave. The default is 1.
        minFreq : float, optional
            Central frequency of the first band. The default is 2e1.
        maxFreq : float, optional
            Central frequency of the last band. The default is 2e4.
        *args : Tuple
            See Analysis.
        bypassLundeby : bool, optional
            Bypass Lundeby calculation, or not. The default is False.
        suppressWarnings : bool, optional
            Supress Lundeby warnings. The default is True.
        ircut : float, optional
            Cut the IR and throw away the silence tail. The default is None.
        **kwargs : Dict
            See Analysis.

        Returns
        -------
        None.

        """
        _ir = ir.IR if type(ir) == ImpulsiveResponse else ir
        minBand = freq_to_band(minFreq, nthOct, 1000, 10)
        maxBand = freq_to_band(maxFreq, nthOct, 1000, 10)
        nbands = maxBand - minBand + 1
        super().__init__('mixed', nthOct, minFreq, maxFreq, nbands*[0], *args, **kwargs)
        self.ir = crop_IR(_ir, ircut)
        fs = ir.samplingRate
        of = OctFilter(order=4,
                       nthOct=self.nthOct,
                       samplingRate=fs,
                       minFreq=self.minBand,
                       maxFreq=self.maxBand,
                       refFreq=1000,
                       base=10)
        filtir, = of(self.ir)
        self._params = self.estimate_energy_parameters(filtir, self.bands, bypassLundeby, suppressWarnings)
        return

    @staticmethod
    def estimate_energy_parameters(filtir: SignalObj, bands:np.ndarray,
                                   bypassLundeby: bool = False,
                                   suppressWarnings: bool = False):
        """
        Estimate the Impulse Response energy parameters.

        Parameters
        ----------
        bypassLundeby : bool
            Whether to bypass calculation of Lundeby IR improvements or not. The default is False.
        suppressWarnings : bool
            If supress warnings about IR quality and the bypassing of Lundeby calculations. The default is False.

        Returns
        -------
        params : Dict[str, np.ndarray]
            A dict with parameters by name.

        """
        listEDC = []
        for ch in range(filtir.numChannels):
            signal = filtir[ch]
            band = bands[ch]
            timeSignal = signal.timeSignal[:]
            timeVector = signal.timeVector[:]
            samplingRate = signal.samplingRate
            numSamples = signal.numSamples
            numChannels = signal.numChannels
            timeLength = signal.timeLength
            energyDecay, energyVector, lundebyParams = \
                energy_decay_calculation(band,
                                         timeSignal,
                                         timeVector,
                                         samplingRate,
                                         numSamples,
                                         numChannels,
                                         timeLength,
                                         bypassLundeby,
                                         suppressWarnings=suppressWarnings)
            listEDC.append([energyDecay, energyVector])
        fs = filtir.samplingRate
        params = {}
        params['rms'] = filtir.rms()
        params['SPL'] = filtir.spl()
        sqrIR = filtir.timeSignal**2
        params['D50'] = definition(sqrIR, fs)
        params['C80'] = clarity(sqrIR, fs)
        params['Ts'] = central_time(sqrIR, filtir.timeVector)
        params['STearly'] = st_early(sqrIR, fs)
        params['STlate'] = st_late(sqrIR, fs)
        params['EDT'] = reverberation_time('EDT', listEDC)
        params['T20'] = reverberation_time(20, listEDC)
        params['T30'] = reverberation_time(30, listEDC)
        # self._params['BR'], self._params['TR'] = timbre_ratios(self.T20)
        return params

    @property
    def parameters(self):
        """List of parameters names."""
        return tuple(self._params.keys())

    @property
    def rms(self):
        """Effective IR amplitude by frequency `band`."""
        return self._params['rms']

    @property
    def SPL(self):
        """Equivalent IR level by frequency `band`."""
        return self._params['SPL']

    @property
    def D50(self):
        """Room Definition by frequency `band`."""
        return self._params['D50']

    @property
    def C80(self):
        """Effective IR amplitude, by frequency `band`."""
        return self._params['C80']

    @property
    def Ts(self):
        """Central Time by frequency `band`."""
        return self._params['Ts']

    @property
    def STearly(self):
        """Early energy distribution by frequency `band`."""
        return self._params['STearly']

    @property
    def STlate(self):
        """Late energy distribution by frequency `band`."""
        return self._params['STlate']

    @property
    def EDT(self):
        """Early Decay Time by frequency `band`."""
        return self._params['EDT']

    @property
    def T20(self):
        """Reverberation time with 20 dB decay, by frequency `band`."""
        return self._params['T20']

    @property
    def T30(self):
        """Reverberation time with 30 dB decay, by frequency `band`."""
        return self._params['T30']

    # @property
    # def BR(self):
    #     """Reverberation time with 30 dB decay, by frequency `band`."""
    #     return self._params['BR']

    # @property
    # def TR(self):
    #     """Reverberation time with 30 dB decay, by frequency `band`."""
    #     return self._params['TR']

    def plot_param(self, name: str, **kwargs):
        """
        Plot a chart with the parameter passed in as `name`.


        Parameters
        ----------
        name : str
            Room parameter name, e.g. `'T20' | 'C80' | 'SPL'`, etc.
        kwargs: Dict
            All kwargs accepted by `Analysis.plot_bar`.

        Returns
        -------
        f : matplotlib.Figure
            The figure of the plot chart.

        """
        self.data = self._params[name]
        self.ylabel = name + ' of room IR'
        f = self.plot(**kwargs)
        self.data = np.zeros(self.bands.shape)
        return f

    def plot_rms(self, **kwargs):
        """Plot a chart for the impulse response's `rms` by frequency `bands`."""
        return self.plot_param('rms', **kwargs)

    def plot_SPL(self, **kwargs):
        """Plot a chart for the impulse response's `SPL` by frequency `bands`."""
        return self.plot_param('SPL', **kwargs)

    def plot_C80(self, **kwargs):
        """Plot a chart for the impulse response's `C80` by frequency `bands`."""
        return self.plot_param('C80', **kwargs)

    def plot_D50(self, **kwargs):
        """Plot a chart for the impulse response's `D50` by frequency `bands`."""
        return self.plot_param('D50', **kwargs)

    def plot_T20(self, **kwargs):
        """Plot a chart for the impulse response's `T20` by frequency `bands`."""
        return self.plot_param('T20', **kwargs)

    def plot_T30(self, **kwargs):
        """Plot a chart for the impulse response's `T30` by frequency `bands`."""
        return self.plot_param('T30', **kwargs)

    def plot_Ts(self, **kwargs):
        """Plot a chart for the impulse response's `Ts` by frequency `bands`."""
        return self.plot_param('Ts', **kwargs)

    def plot_EDT(self, **kwargs):
        """Plot a chart for the impulse response's `EDT` by frequency `bands`."""
        return self.plot_param('EDT', **kwargs)

    def plot_STearly(self, **kwargs):
        """Plot a chart for the impulse response's `STearly` by frequency `bands`."""
        return self.plot_param('STearly', **kwargs)

    def plot_STlate(self, **kwargs):
        """Plot a chart for the impulse response's `STlate` by frequency `bands`."""
        return self.plot_param('STlate', **kwargs)

    # def plot_BR(self):
    #     """Plot a chart for the impulse response's `BR` by frequency `bands`."""
    #     return self.plot_param('BR')

    # def plot_TR(self):
    #     """Plot a chart for the impulse response's `TR` by frequency `bands`."""
    #     return self.plot_param('TR')




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
    return result[0]


# @njit
def _level_profile(timeSignal, samplingRate,
                   numSamples, numChannels, blockSamples=None):
    """Get h(t) in octave bands and do the local time averaging in nblocks. Returns h^2_averaged(block)."""
    def mean_squared(x):
        return np.mean(x**2)

    if blockSamples is None:
        blockSamples = 100
    nblocks = int(numSamples // blockSamples)
    profile = np.zeros((nblocks, numChannels), dtype=np.float32)
    timeStamp = np.zeros((nblocks, 1))

    for ch in range(numChannels):
        # if numChannels == 1:
        #     tmp = timeSignal
        # else:
        tmp = timeSignal[:, ch]
        for idx in range(nblocks):
            profile[idx, ch] = mean_squared(tmp[:blockSamples])
            timeStamp[idx, 0] = idx*blockSamples/samplingRate
            tmp = tmp[blockSamples:]
    return profile, timeStamp


# @njit
def _start_sample_ISO3382(timeSignal, threshold) -> np.ndarray:
    squaredIR = timeSignal**2
    # assume the last 10% of the IR is noise, and calculate its noise level
    last10Idx = -int(len(squaredIR)//10)
    noiseLevel = np.mean(squaredIR[last10Idx:])
    # get the maximum of the signal, that is the assumed IR peak
    max_val = np.max(squaredIR)
    max_idx = np.argmax(squaredIR)
    # check if the SNR is enough to assume that the signal is an IR. If not,
    # the signal is probably not an IR, so it starts at sample 1
    idxNoShift = np.asarray([max_val < 100*noiseLevel or
                             max_idx > int(0.9*squaredIR.shape[0])])
    # less than 20dB SNR or in the "noisy" part
    if idxNoShift.any():
        print("noiseLevelCheck: The SNR too bad or this is not an " +
              "impulse response.")
        return 0
    # find the first sample that lies under the given threshold
    threshold = abs(threshold)
    startSample = 1
#    # TODO - envelope mar/pdi - check!
#    if idxNoShift:
#        print("Something wrong!")
#        return
    # if maximum lies on the first point, then there is no point in searching
    # for the beginning of the IR. Just return this position.
    if max_idx > 0:
        abs_dat = 10*np.log10(squaredIR[:max_idx]) \
                  - 10.*np.log10(max_val)
        thresholdNotOk = True
        thresholdShift = 0
        while thresholdNotOk:
            if len(np.where(abs_dat < (-threshold+thresholdShift))[0]) > 0:
                lastBelowThreshold = \
                    np.where(abs_dat < (-threshold+thresholdShift))[0][-1]
                thresholdNotOk = False
            else:
                thresholdShift += 1
        if thresholdShift > 0:
            print("_start_sample_ISO3382: 20 dB threshold too high. " +
                  "Decreasing it.")
        if lastBelowThreshold > 0:
            startSample = lastBelowThreshold
        else:
            startSample = 1
    return startSample


# @njit
def _circular_time_shift(timeSignal, threshold=20):
    # find the first sample where inputSignal level > 20 dB or > bgNoise level
    startSample = _start_sample_ISO3382(timeSignal, threshold)
    newTimeSignal = timeSignal[startSample:]
    return (newTimeSignal, startSample)


# @njit
def _Lundeby_correction(band, timeSignal, samplingRate, numSamples,
                        numChannels, timeLength, suppressWarnings=True):
    returnTuple = (np.float32(0), np.float32(0), np.int32(0), np.float32(0))
    timeSignal, sampleShift = _circular_time_shift(timeSignal)
    if sampleShift is None:
        return returnTuple

    numSamples -= sampleShift  # discount shifted samples
    numParts = 5  # number of parts per 10 dB decay. N = any([3, 10])
    dBtoNoise = 7  # stop point 10 dB above first estimated background noise
    useDynRange = 15  # dynamic range

    # Window length - 10 to 50 ms, longer periods for lower frequencies and vice versa
    repeat = True
    i = 0
    winTimeLength = 0.01
    while repeat: # loop to find proper winTimeLength
        winTimeLength = winTimeLength + 0.01*i
        # 1) local time average:
        blockSamples = int(winTimeLength * samplingRate)
        timeWinData, timeVecWin = _level_profile(timeSignal, samplingRate,
                                                  numSamples, numChannels,
                                                  blockSamples)

        # 2) estimate noise from h^2_averaged(block):
        bgNoiseLevel = 10 * \
                       np.log10(
                                np.mean(timeWinData[-int(timeWinData.size/10):]))

        # 3) Calculate premilinar slope
        startIdx = np.argmax(np.abs(timeWinData/np.max(np.abs(timeWinData))))
        stopIdx = startIdx + np.where(10*np.log10(timeWinData[startIdx+1:])
                                      >= bgNoiseLevel + dBtoNoise)[0][-1]
        dynRange = 10*np.log10(timeWinData[stopIdx]) \
            - 10*np.log10(timeWinData[startIdx])
        if (stopIdx == startIdx) or (dynRange > -5)[0]:
            if not suppressWarnings:
                print(band, "[Hz] band: SNR too low for the preliminar slope",
                  "calculation.")
            # return returnTuple

        # X*c = EDC (energy decaying curve)
        X = np.ones((stopIdx-startIdx, 2), dtype=np.float32)
        X[:, 1] = timeVecWin[startIdx:stopIdx, 0]
        c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]),
                            rcond=-1)[0]

        if (c[1] == 0)[0] or np.isnan(c).any():
            if not suppressWarnings:
                print(band, "[Hz] band: regression failed. T would be inf.")
            # return returnTuple

        # 4) preliminary intersection
        crossingPoint = (bgNoiseLevel - c[0]) / c[1]  # [s]
        if (crossingPoint > 2*(timeLength + sampleShift/samplingRate))[0]:
            if not suppressWarnings:
                print(band, "[Hz] band: preliminary intersection point between",
                      "bgNoiseLevel and the decay slope greater than signal length.")
            # return returnTuple

        # 5) new local time interval length
        nBlocksInDecay = numParts * dynRange[0] / -10

        dynRangeTime = timeVecWin[stopIdx] - timeVecWin[startIdx]
        blockSamples = int(samplingRate * dynRangeTime[0] / nBlocksInDecay)

        # 6) average
        timeWinData, timeVecWin = _level_profile(timeSignal, samplingRate,
                                                 numSamples, numChannels,
                                                 blockSamples)

        oldCrossingPoint = 11+crossingPoint  # arbitrary higher value to enter loop
        loopCounter = 0

        while (np.abs(oldCrossingPoint - crossingPoint) > 0.001)[0]:
            # 7) estimate background noise level (BGL)
            bgNoiseMargin = 7
            idxLast10Percent = int(len(timeWinData)-(len(timeWinData)//10))
            bgStartTime = crossingPoint - bgNoiseMargin/c[1]
            if (bgStartTime > timeVecWin[-1:][0])[0]:
                idx10dBDecayBelowCrossPoint = len(timeVecWin)-1
            else:
                idx10dBDecayBelowCrossPoint = \
                    np.where(timeVecWin >= bgStartTime)[0][0]
            BGL = np.mean(timeWinData[np.min(
                    np.array([idxLast10Percent,
                              idx10dBDecayBelowCrossPoint])):])
            bgNoiseLevel = 10*np.log10(BGL)

            # 8) estimate late decay slope
            stopTime = (bgNoiseLevel + dBtoNoise - c[0])/c[1]
            if (stopTime > timeVecWin[-1])[0]:
                stopIdx = 0
            else:
                stopIdx = int(np.where(timeVecWin >= stopTime)[0][0])

            startTime = (bgNoiseLevel + dBtoNoise + useDynRange - c[0])/c[1]
            if (startTime < timeVecWin[0])[0]:
                startIdx = 0
            else:
                startIdx = int(np.where(timeVecWin <= startTime)[0][0])

            lateDynRange = np.abs(10*np.log10(timeWinData[stopIdx]) \
                - 10*np.log10(timeWinData[startIdx]))

            # where returns empty
            if stopIdx == startIdx or (lateDynRange < useDynRange)[0]:
                if not suppressWarnings:
                    print(band, "[Hz] band: SNR for the Lundeby late decay slope too",
                        "low. Skipping!")
                # c[1] = np.inf
                c[1] = 0
                i += 1
                break

            X = np.ones((stopIdx-startIdx, 2), dtype=np.float32)
            X[:, 1] = timeVecWin[startIdx:stopIdx, 0]
            c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]),
                                rcond=-1)[0]

            if (c[1] >= 0)[0]:
                if not suppressWarnings:
                    print(band, "[Hz] band: regression did not work, T -> inf.",
                        "Setting slope to 0!")
                # c[1] = np.inf
                c[1] = 0
                i += 1
                break

            # 9) find crosspoint
            oldCrossingPoint = crossingPoint
            crossingPoint = (bgNoiseLevel - c[0]) / c[1]

            loopCounter += 1
            if loopCounter > 30:
                if not suppressWarnings:
                    print(band, "[Hz] band: more than 30 iterations on regression.",
                        "Canceling!")
                break

        interIdx = crossingPoint * samplingRate # [sample]
        i += i
        if c[1][0] != 0:
            repeat = False
        if i > 5:
            if not suppressWarnings:
                print(band, "[Hz] band: too many iterations to find winTimeLength.", "Canceling!")
            return returnTuple

    return c[0][0], c[1][0], np.int32(interIdx[0]), BGL

# @njit
def energy_decay_calculation(band, timeSignal, timeVector, samplingRate,
                             numSamples, numChannels, timeLength, bypassLundeby, suppressWarnings=True):
    """Calculate the Energy Decay Curve."""
    if not bypassLundeby:
        lundebyParams = \
            _Lundeby_correction(band,
                                timeSignal,
                                samplingRate,
                                numSamples,
                                numChannels,
                                timeLength,
                                suppressWarnings=suppressWarnings)
        _, c1, interIdx, BGL = lundebyParams
        lateRT = -60/c1 if c1 != 0 else 0
    else:
        interIdx = 0
        lateRT = 1

    if interIdx == 0:
        interIdx = -1

    truncatedTimeSignal = timeSignal[:interIdx, 0]
    truncatedTimeVector = timeVector[:interIdx]

    if lateRT != 0.0:
        if not bypassLundeby:
            C = samplingRate*BGL*lateRT/(6*np.log(10))
        else:
            C = 0
        sqrInv = truncatedTimeSignal[::-1]**2
        energyDecayFull = np.cumsum(sqrInv)[::-1] + C
        energyDecay = energyDecayFull/energyDecayFull[0]
    else:
        if not suppressWarnings:
            print(band, "[Hz] band: could not estimate C factor")
        C = 0
        energyDecay = np.zeros(truncatedTimeVector.size)
    return (energyDecay, truncatedTimeVector, lundebyParams)

def cumulative_integration(inputSignal,
                           bypassLundeby,
                           plotLundebyResults,
                           suppressWarnings=True,
                           **kwargs):
    """Cumulative integration with proper corrections."""

    def plot_lundeby():
        c0, c1, interIdx, BGL = lundebyParams
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_axes([0.08, 0.15, 0.75, 0.8], polar=False,
                            projection='rectilinear', xscale='linear')
        line = c1*timeVector + c0
        ax.plot(timeVector, 10*np.log10(timeSignal**2), label='IR')
        ax.axhline(y=10*np.log10(BGL), color='#1f77b4', label='BG Noise', c='red')
        ax.plot(timeVector, line,label='Late slope', c='black')
        ax.axvline(x=interIdx/samplingRate, label='Truncation point', c='green')
        ax.grid()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude [dBFS]')
        plt.title('{0:.0f} [Hz]'.format(band))
        ax.legend(loc='best', shadow=True, fontsize='x-large')

    timeSignal = inputSignal.timeSignal[:]
    # Substituted by SignalObj.crop in analyse function
    # timeSignal, sampleShift = _circular_time_shift(timeSignal)
    # del sampleShift
    hSignal = SignalObj(timeSignal,
                        inputSignal.lengthDomain,
                        inputSignal.samplingRate)
    hSignal = _filter(hSignal, **kwargs)
    bands = FOF(nthOct=kwargs['nthOct'],
                freqRange=[kwargs['minFreq'], kwargs['maxFreq']])[:,1]
    listEDC = []
    for ch in range(hSignal.numChannels):
        signal = hSignal[ch]
        band = bands[ch]
        timeSignal = cp.copy(signal.timeSignal[:])
        timeVector = signal.timeVector[:]
        samplingRate = signal.samplingRate
        numSamples = signal.numSamples
        numChannels = signal.numChannels
        timeLength = signal.timeLength
        energyDecay, energyVector, lundebyParams = \
            energy_decay_calculation(band,
                                     timeSignal,
                                     timeVector,
                                     samplingRate,
                                     numSamples,
                                     numChannels,
                                     timeLength,
                                     bypassLundeby,
                                     suppressWarnings=suppressWarnings)
        listEDC.append((energyDecay, energyVector))
        if plotLundebyResults:  # Placed here because Numba can't handle plots.
            # plot_lundeby(band, timeVector, timeSignal,  samplingRate,
            #             lundebyParams)
            plot_lundeby()
    return listEDC

# @njit
def reverb_time_regression(energyDecay, energyVector, upperLim, lowerLim):
    """Interpolate the EDT to get the reverberation time."""
    if not np.any(energyDecay):
        return 0
    first = np.where(10*np.log10(energyDecay) >= upperLim)[0][-1]
    last = np.where(10*np.log10(energyDecay) >= lowerLim)[0][-1]
    if last <= first:
        # return np.nan
        return 0
    X = np.ones((last-first, 2))
    X[:, 1] = energyVector[first:last]
    c = np.linalg.lstsq(X, 10*np.log10(energyDecay[first:last]), rcond=-1)[0]
    return -60/c[1]


def reverberation_time(decay, listEDC):
    """Call the reverberation time regression."""
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
    RT = []
    for ED in listEDC:
        edc, edv = ED
        RT.append(reverb_time_regression(edc, edv, y1, y2))
    return np.array(RT, dtype='float32')


def G_Lpe(IR, nthOct, minFreq, maxFreq, IREndManualCut=None):
    """
    Calculate the energy level from the room impulsive response.

    Reference:
        Christensen, C. L.; Rindel, J. H. APPLYING IN-SITU RECALIBRATION FOR
        SOUND STRENGTH MEASUREMENTS IN AUDITORIA.

    :param IR: one channel impulsive response
    :type IR: ImpulsiveResponse

    :param nthOct: number of fractions per octave
    :type nthOct: int

    :param minFreq: analysis inferior frequency limit
    :type minFreq: float

    :param maxFreq: analysis superior frequency limit
    :type maxFreq: float

    :return: Analysis object with the calculated parameter
    :rtype: Analysis
    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    # firstChNum = IR.systemSignal.channels.mapping[0]
    # if not IR.systemSignal.channels[firstChNum].calibCheck:
    #     raise ValueError("'IR' must be a calibrated ImpulsiveResponse")
    if isinstance(IR, SignalObj):
        SigObj = cp.copy(IR)
    elif isinstance(IR, ImpulsiveResponse):
        SigObj = cp.copy(IR.systemSignal)
    else:
        raise TypeError("'IR' must be an ImpulsiveResponse or SignalObj.")
    # Cutting the IR
    if IREndManualCut is not None:
        SigObj.crop(0, IREndManualCut)
    timeSignal, _ = _circular_time_shift(SigObj.timeSignal[:,0])
    # Bands filtering
    # hSignal = SignalObj(SigObj.timeSignal[:,0],
    hSignal = SignalObj(timeSignal,
                        SigObj.lengthDomain,
                        SigObj.samplingRate)
    hSignal = _filter(signal=hSignal, nthOct=nthOct, minFreq=minFreq,
                      maxFreq=maxFreq)
    bands = FOF(nthOct=nthOct,
                freqRange=[minFreq,maxFreq])[:,1]
    Lpe = []
    for chIndex in range(hSignal.numChannels):
        Lpe.append(
            10*np.log10(np.trapz(y=hSignal.timeSignal[:,chIndex]**2/(2e-5**2),
                                 x=hSignal.timeVector)))
    LpeAnal = Analysis(anType='mixed', nthOct=nthOct, minBand=float(bands[0]),
                       maxBand=float(bands[-1]), data=Lpe,
                       comment='h**2 energy level')
    LpeAnal.creation_name = creation_name
    return LpeAnal


def G_Lps(IR, nthOct, minFreq, maxFreq):
    # TODO: Fix documentation format
    """G_Lps

    Calculates the recalibration level, for both in-situ and
    reverberation chamber. Lps is applied for G calculation.

    During the recalibration: source height and mic heigth must be >= 1 [m],
    while the distance between source and mic must be <= 1 [m]. The distances
    must be the same for in-situ and reverberation chamber measurements.

    Reference:
        Christensen, C. L.; Rindel, J. H. APPLYING IN-SITU RECALIBRATION FOR
        SOUND STRENGTH MEASUREMENTS IN AUDITORIA.

    :param IR: one channel impulsive response
    :type IR: ImpulsiveResponse

    :param nthOct: number of fractions per octave
    :type nthOct: int

    :param minFreq: analysis inferior frequency limit
    :type minFreq: float

    :param maxFreq: analysis superior frequency limit
    :type maxFreq: float

    :return: Analysis object with the calculated parameter
    :rtype: Analysis
    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    # firstChNum = IR.systemSignal.channels.mapping[0]
    # if not IR.systemSignal.channels[firstChNum].calibCheck:
    #     raise ValueError("'IR' must be a calibrated ImpulsiveResponse")
    if isinstance(IR, SignalObj):
        SigObj = IR
    elif isinstance(IR, ImpulsiveResponse):
        SigObj = IR.systemSignal
    else:
        raise TypeError("'IR' must be an ImpulsiveResponse or SignalObj.")
    # Windowing the IR
    # dBtoOnSet = 20
    # dBIR = 10*np.log10((SigObj.timeSignal[:,0]**2)/((2e-5)**2))
    # windowStart = np.where(dBIR > (max(dBIR) - dBtoOnSet))[0][0]

    broadBandTimeSignal = cp.copy(SigObj.timeSignal[:,0])
    broadBandTimeSignalNoStart, sampleShift = \
        _circular_time_shift(broadBandTimeSignal)
    windowLength = 0.0032 # [s]
    windowEnd = int(windowLength*SigObj.samplingRate)


    hSignal = SignalObj(broadBandTimeSignalNoStart[:windowEnd],
    # hSignal = SignalObj(timeSignal,
                        SigObj.lengthDomain,
                        SigObj.samplingRate)
    hSignal = _filter(signal=hSignal, nthOct=nthOct, minFreq=minFreq,
                      maxFreq=maxFreq)
    bands = FOF(nthOct=nthOct,
                freqRange=[minFreq,maxFreq])[:,1]
    Lps = []
    for chIndex in range(hSignal.numChannels):
        timeSignal = cp.copy(hSignal.timeSignal[:,chIndex])
        # timeSignalNoStart, sampleShift = _circular_time_shift(timeSignal)
        # windowLength = 0.0032 # [s]
        # windowEnd = int(windowLength*SigObj.samplingRate)

        Lps.append(
            # 10*np.log10(np.trapz(y=timeSignalNoStart[:windowEnd]**2/(2e-5**2),
            10*np.log10(np.trapz(y=timeSignal**2/(2e-5**2),
                                #  x=hSignal.timeVector[sampleShift:sampleShift+windowEnd])))
                                 x=hSignal.timeVector)))
    LpsAnal = Analysis(anType='mixed', nthOct=nthOct, minBand=float(bands[0]),
                       maxBand=float(bands[-1]), data=Lps,
                       comment='Source recalibration method IR')
    LpsAnal.creation_name = creation_name
    LpsAnal.windowLimits = ((sampleShift)/SigObj.samplingRate,
                            (sampleShift+windowEnd)/SigObj.samplingRate)
    # Plot IR cutting
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_axes([0.08, 0.15, 0.75, 0.8], polar=False,
    #                         projection='rectilinear', xscale='linear')
    # ax.plot(SigObj.timeVector, 10*np.log10(SigObj.timeSignal**2/2e-5**2))
    # ax.axvline(x=(sampleShift)/SigObj.samplingRate, linewidth=4, color='k')
    # ax.axvline(x=(sampleShift+windowEnd)/SigObj.samplingRate, linewidth=4, color='k')
    # ax.set_xlim([(sampleShift-100)/SigObj.samplingRate, (sampleShift+windowEnd+100)/SigObj.samplingRate])
    return LpsAnal


def strength_factor(Lpe, Lpe_revCh, V_revCh, T_revCh, Lps_revCh, Lps_inSitu):
    """
    Calculate strength factor (G) for theaters and big audience intended places.

    Reference:
        Christensen, C. L.; Rindel, J. H. APPLYING IN-SITU RECALIBRATION FOR
        SOUND STRENGTH MEASUREMENTS IN AUDITORIA.
    """
    # TO DO: docs
    S0 = 1 # [m2]

    bands = T_revCh.bands
    nthOct = T_revCh.nthOct
    terms = []
    for bandData in T_revCh.data:
        if bandData == 0:
            terms.append(0)
        else:
            term = (V_revCh * 0.16) / (bandData * S0)
            terms.append(term)
    terms = [10*np.log10(term) if term != 0 else 0 for term in terms]

    revChTerm = Analysis(anType='mixed', nthOct=nthOct, minBand=float(bands[0]),
                         maxBand=float(bands[-1]), data=terms)
    Lpe.anType = 'mixed'
    Lpe_revCh.anType = 'mixed'
    Lps_revCh.anType = 'mixed'
    Lps_inSitu.anType = 'mixed'
    G = Lpe - Lpe_revCh - revChTerm + 37 \
        + Lps_revCh - Lps_inSitu
    G.anType = 'G'
    return G


def definition(sqrIR: np.ndarray, fs: int, t: int = 50) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.
    t_ms : int, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    definition : np.ndarray
        The room "Definition" parameter, in percentage [%].

    """
    t_ms = t * fs // 1000
    sumSIRt = sqrIR.sum(axis=0)  # total sum of squared IR
    sumSIRi = sqrIR[:t_ms].sum(axis=0)  # sum of initial portion of squared IR
    definition = np.round(100 * (sumSIRi / sumSIRt), 2)  # [%]
    return definition


def clarity(sqrIR: np.ndarray, fs: int, t: int = 80) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.
    t_ms : int, optional
        DESCRIPTION. The default is 80.

    Returns
    -------
    clarity : np.ndarray
        The room "Clarity" parameter, in decibel [dB].

    """
    t_ms = t * fs // 1000
    sumSIRi = sqrIR[:t_ms].sum(axis=0)  # sum of initial portion of squared IR
    sumSIRe = sqrIR[t_ms:].sum(axis=0)  # sum of ending portion of squared IR
    clarity = np.round(10 * np.log10(sumSIRi / sumSIRe), 2)  # [dB]
    return clarity


def central_time(sqrIR: np.ndarray, tstamp: np.ndarray) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        Squared room impulsive response.
    tstamp : np.ndarray
        Time stamps of each IR sample.

    Returns
    -------
    central_time : np.ndarray
        The time instant that balance of energy is equal before and after it.

    """
    sumSIR = sqrIR.sum(axis=0)
    sumTSIR = (tstamp[:, None] * sqrIR).sum(axis=0)
    central_time = (sumTSIR / sumSIR) * 1000  # milisseconds
    return central_time


def st_early(sqrIR: np.ndarray, fs: int) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.

    Returns
    -------
    STearly : np.ndarray
        DESCRIPTION.

    """
    ms = fs / 1000
    sum10ms = sqrIR[:int(10 * ms)].sum(axis=0)
    sum20ms = sqrIR[int(20 * ms):int(100 * ms)].sum(axis=0)
    STearly = 10 * np.log10(sum20ms / sum10ms)
    return np.round(STearly, 4)


def st_late(sqrIR: np.ndarray, fs: int) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.

    Returns
    -------
    STlate : np.ndarray
        DESCRIPTION.

    """
    ms = fs / 1000
    sum10ms = sqrIR[:int(10 * ms)].sum(axis=0)
    sum100ms = sqrIR[int(100 * ms):int(1000 * ms)].sum(axis=0)
    STlate = 10 * np.log10(sum100ms / sum10ms)
    return np.round(STlate, 4)


def crop_IR(SigObj, IREndManualCut):
    """Cut the impulse response at background noise level."""
    timeSignal = cp.copy(SigObj.timeSignal)
    timeVector = SigObj.timeVector
    samplingRate = SigObj.samplingRate
    numSamples = SigObj.numSamples
    # numChannels = SigObj.numChannels
    if SigObj.numChannels > 1:
        print('crop_IR: The provided impulsive response has more than one ' +
              'channel. Cropping based on channel 1.')
    numChannels = 1
    # Cut the end automatically or manual
    if IREndManualCut is None:
        winTimeLength = 0.1  # [s]
        meanSize = 5  # [blocks]
        dBtoReplica = 6  # [dB]
        blockSamples = int(winTimeLength * samplingRate)
        timeWinData, timeVecWin = _level_profile(timeSignal, samplingRate,
                                                numSamples, numChannels,
                                                blockSamples)
        endTimeCut = timeVector[-1]
        for blockIdx, blockAmplitude in enumerate(timeWinData):
            if blockIdx >= meanSize:
                anteriorMean = 10*np.log10( \
                    np.sum(timeWinData[blockIdx-meanSize:blockIdx])/meanSize)
                if 10*np.log10(blockAmplitude) > anteriorMean+dBtoReplica:
                    endTimeCut = timeVecWin[blockIdx-meanSize//2]
                    break
    else:
        endTimeCut = IREndManualCut
    endTimeCutIdx = np.where(timeVector >= endTimeCut)[0][0]
    timeSignal = timeSignal[:endTimeCutIdx]
    # Cut the start automatically
    timeSignal, _ = _circular_time_shift(timeSignal)
    result = SignalObj(timeSignal,
                       'time',
                       samplingRate,
                       signalType='energy')
    return result


def analyse(obj, *params,
            bypassLundeby=False,
            plotLundebyResults=False,
            suppressWarnings=False,
            IREndManualCut=None, **kwargs):
    """
    Room analysis over a single SignalObj.

    Receives an one channel SignalObj or ImpulsiveResponse and calculate the
    room acoustic parameters especified in the positional input arguments.
    Calculates reverberation time, definition and clarity.

    The method for strength factor calculation implies in many input parameters
    and specific procedures, as the sound source's power estimation.
    The pytta.roomir app was designed aiming to support this room parameter
    measurement. For further information check pytta.roomir's and
    pytta.rooms.strength_factor's docstrings.

    Input arguments (default), (type):
    -----------------------------------

        * obj (), (SignalObj | ImpulsiveResponse):
            one channel impulsive response

        * non-keyworded argument pairs:
            Pair for 'RT' (reverberation time):

                - RTdecay (20), (int):
                    Decay interval for RT calculation. e.g. 20

            Pair for 'C' (clarity):  # TODO

                - Cparam (50), (int):
                    ...

            Pair for 'D' (definition):  # TODO

                - Dparam (50), (int):
                    ...

        * nthOct (), (int):
            Number of bands per octave;

        * minFreq (), (int | float):
            Analysis' inferior frequency limit;

        * maxFreq (), (int | float):
            Analysis' superior frequency limit;

        * bypassLundeby (false), (bool):
            Bypass lundeby correction

        * plotLundebyResults (false), (bool):
            Plot the Lundeby correction parameters;

        * suppressWarnings (false), (bool):
            Suppress the warnings from the Lundeby correction;


    Return (type):
    --------------


        * Analyses (Analysis | list):
            Analysis object with the calculated parameter or a list of
            Analyses for more than one parameter.

    Usage example:

        >>> myRT = pytta.rooms.analyse(IR,
                                       'RT', 20',
                                       'C', 50,
                                       'D', 80,
                                       nthOct=3,
                                       minFreq=100,
                                       maxFreq=10000)

    For more tips check the examples folder.

    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
    # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    if not isinstance(obj, SignalObj) and not isinstance(obj, ImpulsiveResponse):
        raise TypeError("'obj' must be an one channel SignalObj or" +
                        " ImpulsiveResponse.")
    if isinstance(obj, ImpulsiveResponse):
        SigObj = obj.systemSignal
    else:
        SigObj = obj

    if SigObj.numChannels > 1:
        raise TypeError("'obj' can't contain more than one channel.")
    # samplingRate = SigObj.samplingRate

    SigObj = crop_IR(SigObj, IREndManualCut)

    calcEDC = False
    result = []

    for param in params:
        if param in ['RT']:  # 'C', 'D']:
            calcEDC = True
            break

    if calcEDC:
        listEDC = cumulative_integration(SigObj,
                                         bypassLundeby,
                                         plotLundebyResults,
                                         suppressWarnings=suppressWarnings,
                                         **kwargs)

    if 'RT' in params:
        RTdecay = params[params.index('RT')+1]
        if not isinstance(RTdecay,(int)):
            RTdecay = 20
        nthOct = kwargs['nthOct']
        RT = reverberation_time(RTdecay, listEDC)
        RTtemp = Analysis(anType='RT', nthOct=nthOct,
                          minBand=kwargs['minFreq'],
                          maxBand=kwargs['maxFreq'],
                          data=RT)
        RTtemp.creation_name = creation_name
        result.append(RTtemp)

    # if 'C' in params:
    #     Cparam = params[params.index('C')+1]
    #     if not isinstance(Cparam,(int)):
    #         Cparam = 50
    #     Ctemp = None
    #     result.append(Ctemp)

    # if 'D' in params:
    #     Dparam = params[params.index('D')+1]
    #     if not isinstance(Dparam,(int)):
    #         Dparam = 50
    #     Dtemp = None
    #     result.append(Dtemp)

    if len(result) == 1:
        result = result[0]

    return result

