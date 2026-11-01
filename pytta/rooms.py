# -*- coding: utf-8 -*-

"""
DEPRECATED
----------
    Being replaced by class pytta.RoomAnalysis on version 0.1.1.

Calculations compliant to ISO 3382-1 to obtain room acoustic parameters.

It has an implementation of Lundeby et al. [1] algorithm
to estimate the correction factor for the cumulative integral, as suggested
by the ISO 3382-1.

Use this module through the function 'analyse', which receives an one channel
SignalObj or ImpulsiveResponse and calculate the room acoustic parameters
especified in the positional input arguments. For more information check
pytta.rooms.analyse's documentation.

Available functions:

    >>> pytta.rooms.Analyse(SignalObj, ...)
    >>> pytta.rooms.strength_factor(...)
    >>> pytta.rooms.G_Lpe
    >>> pytta.rooms.G_Lps
    >>> pytta.rooms.image_source_ir(dimensions, source, receiver, fs, duration)

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
from pytta.classes.analysis import _circular_time_shift, _filter, crop_IR, \
    cumulative_integration, reverberation_time
import traceback
import copy as cp
from warnings import warn


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

    During the recalibration: source height and mic height must be >= 1 [m],
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

def analyse(obj, *params,
            bypassLundeby=False,
            plotLundebyResults=False,
            suppressWarnings=False,
            IREndManualCut=None, **kwargs):
    """
    DEPRECATED
    ----------
        Being replaced by class pytta.RoomAnalysis on version 0.1.1.
        
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
    
    warn(DeprecationWarning("Function 'pytta.Analyse' is DEPRECATED and " +
                            "being replaced by the class pytta.RoomAnalysis" +
                            " on version 0.1.1"))

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


def image_source_ir(dimensions, source, receiver, samplingRate, duration,
                    reflection=1., c=343.2, half_width=8):
    """
    Impulse response of a rectangular room by the image source method of
    Allen and Berkley (1979), as ita_roomacoustics_imagesources: the free
    field Green's function delta(t - d/c) / (4 pi d) of every image source
    within duration, weighted by the reflection factors of the walls it
    goes through.

    Parameters
    ----------
    dimensions, source, receiver : array_like
        (Lx, Ly, Lz) and the positions (m), with 0 <= x <= Lx.
    samplingRate : int
    duration : float
        Length of the response (s); image sources farther than c duration
        are left out.
    reflection : float or array_like
        Pressure reflection factor of every wall, or six of them for the
        walls x = 0, x = Lx, y = 0, y = Ly, z = 0, z = Lz (frequency and
        angle independent; ita_roomacoustics_imagesources derives them
        from a wall impedance Z as (Z - rho c) / (Z + rho c)). Default 1.
    half_width : int
        Half length in samples of the Hann-windowed sinc that places each
        arrival between samples. Default 8.

    Returns
    -------
    SignalObj
        The response; convolved with rho dQ/dt of the source (Q the volume
        velocity) it gives the pressure.
    """
    from pytta.classes import SignalObj
    L = np.asarray(dimensions, dtype=float)
    r0, r = np.asarray(source, dtype=float), np.asarray(receiver, dtype=float)
    if np.any(r0 < 0) or np.any(r0 > L) or np.any(r < 0) or np.any(r > L):
        raise ValueError('source and receiver must lie inside the room.')
    beta = np.broadcast_to(np.asarray(reflection, dtype=float), (6,))
    n = int(round(duration * samplingRate))
    out = np.zeros(n + 2 * half_width + 1)
    reach = c * duration
    lmax = [int(np.ceil(reach / (2 * L[i]))) + 1 for i in range(3)]
    k = np.arange(-half_width, half_width + 1)
    # per axis: image coordinates, distances and reflection weights of
    # x = 2 l Lx + (1 - 2u) x0 (u = 0, 1): |l - u| reflections at x = 0 and
    # |l| at x = Lx
    axes = []
    for i in range(3):
        ls = np.arange(-lmax[i], lmax[i] + 1)
        coord, weight = [], []
        for u in (0, 1):
            coord.append(2 * ls * L[i] + (1 - 2 * u) * r0[i] - r[i])
            weight.append(beta[2 * i] ** np.abs(ls - u) * beta[2 * i + 1] ** np.abs(ls))
        axes.append((np.concatenate(coord), np.concatenate(weight)))
    (dx, wx), (dy, wy), (dz, wz) = axes
    d2xy = dx[:, None] ** 2 + dy[None, :] ** 2
    wxy = wx[:, None] * wy[None, :]
    for z, w in zip(dz, wz):
        d = np.sqrt(d2xy + z ** 2).ravel()
        keep = d <= reach
        if not np.any(keep):
            continue
        d, a = d[keep], (wxy.ravel() * w)[keep] / (4 * np.pi * d[keep])
        t = d / c * samplingRate
        i0 = np.floor(t).astype(int)
        frac = t - i0
        x = k[None, :] - frac[:, None]
        kernel = np.sinc(x) * 0.5 * (1 + np.cos(np.pi * x / (half_width + 1)))
        idx = i0[:, None] + k[None, :] + half_width
        np.add.at(out, idx.ravel(), (a[:, None] * kernel).ravel())
    return SignalObj(out[half_width:half_width + n], 'time', samplingRate,
                     signalType='energy')
