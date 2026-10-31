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
    >>> pytta.rooms.air_constants(temperature, humidity, pressure)
    >>> pytta.rooms.convert_room_quantity(value, 'T', 'alpha', volume, surface)
    >>> pytta.rooms.din18041(volume, purpose)

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


def air_constants(temperature=20., humidity=0.5, pressure=101325.):
    """
    Speed of sound, density and characteristic impedance of humid air, as
    ita_constants of the ITA-Toolbox: the gas constant of air with the
    molar concentration of water vapour of ISO 9613-1, c = sqrt(1.4 R T)
    and rho = p / (R T).

    Parameters
    ----------
    temperature : float
        Celsius. Default 20.
    humidity : float
        Relative humidity, 0 to 1. Default 0.5.
    pressure : float
        Static pressure in Pa. Default 101325.

    Returns
    -------
    dict
        'c' (m/s), 'rho' (kg/m^3) and 'z0' (Pa s/m). The power attenuation
        of air, m, is pytta.iso354.power_attenuation.
    """
    if not 0 <= humidity <= 1:
        raise ValueError('humidity is a relative humidity from 0 to 1.')
    T = temperature + 273.15
    p_sat = 101325. * 10 ** (-6.8346 * (273.15 / T) ** 1.261 + 4.6151)
    h = 100 * humidity * p_sat / pressure          # molar concentration, %
    R_dry = 8.31 / 0.0289644                       # J/(kg K)
    R = R_dry / (1 - h / 100 * (1 - R_dry / 461.))
    c = np.sqrt(1.4 * R * T)
    rho = pressure / (R * T)
    return {'c': float(c), 'rho': float(rho), 'z0': float(rho * c)}


_QUANTITIES = ('T', 'alpha', 'R', 'Z')


def convert_room_quantity(value, source, target, volume=None, surface=None,
                          formula='eyring', c=343.2, z0=413.3):
    """
    Conversion between reverberation time T (s), absorption coefficient
    alpha, normal-incidence reflection factor R and surface impedance Z
    (Pa s/m), as ita_convert_RT_alpha_R_Z.

    T and alpha are related by Eyring, T = 24 ln(10) V / (-c S ln(1 -
    alpha)), or Sabine, T = 24 ln(10) V / (c S alpha), in both directions
    (ita_convert_RT_alpha_R_Z goes from T to alpha by Eyring and back by
    Sabine with air absorption); alpha = 1 - |R|^2, R = sqrt(1 - alpha)
    (real, the phase is not known), and Z = z0 (1 + R) / (1 - R).

    Parameters
    ----------
    value : array_like
    source, target : str
        'T', 'alpha', 'R' or 'Z'.
    volume, surface : float
        Room volume (m^3) and surface (m^2), for conversions with T.
    formula : str
        'eyring' (default) or 'sabine'.
    c, z0 : float
        Speed of sound and characteristic impedance of air (see
        air_constants). Defaults for 20 C.
    """
    if source not in _QUANTITIES or target not in _QUANTITIES:
        raise ValueError(f'quantities are {_QUANTITIES}.')
    if formula not in ('eyring', 'sabine'):
        raise ValueError("formula must be 'eyring' or 'sabine'.")
    if 'T' in (source, target) and source != target and (volume is None or surface is None):
        raise ValueError('volume and surface are needed for T.')
    v = np.asarray(value, dtype=complex if source in ('R', 'Z') else float)
    k = 24 * np.log(10) * volume / (c * surface) if volume is not None and surface is not None else None
    # to alpha (or R for Z)
    if source == 'T':
        x = k / v
        alpha = 1 - np.exp(-x) if formula == 'eyring' else x
    elif source == 'alpha':
        alpha = v
    elif source == 'R':
        alpha = 1 - np.abs(v) ** 2
        if target == 'Z':
            return z0 * (1 + v) / (1 - v)
    else:
        R = (v / z0 - 1) / (v / z0 + 1)
        if target == 'R':
            return R
        alpha = 1 - np.abs(R) ** 2
    if target == 'alpha':
        return alpha
    if target == 'T':
        return k / (-np.log(1 - alpha) if formula == 'eyring' else alpha)
    R = np.sqrt(1 - alpha)
    return R if target == 'R' else z0 * (1 + R) / (1 - R)


def din18041(volume, purpose='speech', frequencies=None):
    """
    Recommended reverberation time and its tolerance band by frequency, as
    ita_din18041_reverberation_times (DIN 18041:2004, rooms of groups
    music, speech and education): T_soll = a lg(V) + b, with (a, b) =
    (0.45, 0.07), (0.37, -0.14) and (0.32, -0.17), times the tolerance
    curves of the standard.

    Parameters
    ----------
    volume : float
        Room volume, m^3.
    purpose : str
        'music', 'speech' or 'education'.
    frequencies : array_like, optional
        Frequencies (Hz); by default 1025 points from 0 to 22.05 kHz, as
        ita_din18041_reverberation_times.

    Returns
    -------
    dict
        'T_soll' (s), 'frequency' (Hz), 'T_min' and 'T_max' (s).
    """
    coef = {'music': (0.45, 0.07), 'speech': (0.37, -0.14), 'education': (0.32, -0.17)}
    if purpose not in coef:
        raise ValueError(f'purpose must be one of {tuple(coef)}.')
    f = np.linspace(0, 22050, 1025) if frequencies is None else np.asarray(frequencies, dtype=float)
    a, b = coef[purpose]
    T = a * np.log10(volume) + b
    lg = np.log10(np.maximum(f, 1e-300))
    lo = np.full(f.size, 0.8)
    if purpose == 'music':
        lo[f <= 250] = (np.log10(250) - lg[f <= 250]) * 0.7 + 0.8
        lo[f >= 2000] = (np.log10(2000) - lg[f >= 2000]) * 0.5 + 0.8
        hi = lo * 1.2 / 0.8
        hi[f >= 2000] = 1.2
    else:
        lo[f <= 250] = (lg[f <= 250] - np.log10(250)) * 0.5 + 0.8
        lo[f >= 2000] = (np.log10(2000) - lg[f >= 2000]) * 0.5 + 0.8
        hi = np.full(f.size, 1.2)
    return {'T_soll': float(T), 'frequency': f,
            'T_min': np.maximum(lo * T, 1e-6), 'T_max': np.maximum(hi * T, 1e-6)}
