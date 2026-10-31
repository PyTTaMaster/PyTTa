"""
PyTTa ISO3741 Analysis:
----------------------

This module does some calculations compliant to ISO 3471 in order to obtain
the acoustic power of a source.

Implemented are the mean time-average sound pressure level among the
microphone positions and the sound power level by the direct and by the
comparison methods of ISO 3741:1999, 8.4. In the future we hope the
community keep developing this module in order to provide all the
calculation needed for measurements acordding to this standard.

Available functions:

    >>> pytta.iso3741.Lp_ST()
    >>> pytta.iso3741.sound_power()
    >>> pytta.iso3741.sound_power_comparison()

For further information, check the function specific documentation.

"""

from pytta import SignalObj, OctFilter, Analysis
import numpy as np
from pytta.classes.filter import fractional_octave_frequencies as FOF
import traceback

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

def Lp_ST(sigObjList, nthOct, minFreq, maxFreq, IRManualCut=None):
    """
    Calculate from the provided list of one channel SignalObjs the mean
    one-third-octave band time-averaged sound pressure level in the test room
    with the noise source under test in operation, Lp(ST), and the standard
    deviation, Sm, for the preliminary measurements, located at error property
    of the returned pytta.Analysis.

    Correction for background noise not implemented because of the UFSM
    reverberation chamber's good isolation.

    Parameters (default), (type):
    -----------------------------

        * sigObjList (), (list):
            A list containing SignalObjs of each measurement position.

        * nthOct (), (int):
            The number of fractions per octave;

        * minFreq (), (int | float):
            The exact or approximated start band frequency;

        * maxFreq (), (int | float):
            The exact or approximated stop band frequency;

        * IRManualCut (None), (int | float):
            Optional manual IR cut in seconds.

    Returns:
    ---------

        pytta.Analysis object.

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

    Leqs = []

    for idx, sigObj in enumerate(sigObjList):
        if not sigObj.channels[firstChNum].calibCheck:
            raise ValueError("SignalObj {} must be calibrated.".format(idx+1))
        # Cutting the IR
        if IRManualCut is not None:
            sigObj.crop(0, IRManualCut)
        # Bands filtering
        hSignal = SignalObj(sigObj.timeSignal[:,0],
                            sigObj.lengthDomain,
                            sigObj.samplingRate)
        hSignal = _filter(signal=hSignal, nthOct=nthOct, minFreq=minFreq,
                        maxFreq=maxFreq)
        bands = FOF(nthOct=nthOct,
                    minFreq=minFreq,
                    maxFreq=maxFreq)[:,1]
        Leq = []
        for chIndex in range(hSignal.numChannels):
            Leq.append(
                10*np.log10(np.mean(hSignal.timeSignal[:,chIndex]**2)/
                            (2e-5**2)))
        Leq = Analysis(anType='mixed', nthOct=nthOct,
                           minBand=float(bands[0]),
                           maxBand=float(bands[-1]), data=Leq,
                           comment='Leq')
        Leqs.append(Leq)

    Leq = 0
    for L in Leqs:
        Leq =  L + Leq
    Lp_ST = Leq / len(sigObjList)
    Lp_ST.anType = 'mixed'
    Lp_ST.unit = 'dB'
    Lp_ST.creation_name = creation_name

    # Statistics for Lp_ST
    data = np.vstack([an.data for an in Leqs])
    Sm = []
    for bandIdx in range(data.shape[1]):
        summing = 0
        for idx in range(data.shape[0]):
            summing += \
            (data[idx, bandIdx] - Lp_ST.data[bandIdx])**2 / (data.shape[0] - 1)
        Sm.append(summing**(1/2))

    Lp_ST.error = Sm
    Lp_ST.errorLabel = "Standard deviation"
    return Lp_ST


def sound_power(Lp, T, V, S, f, temperature, pressure):
    """
    Sound power level by the direct method of ISO 3741:1999, 8.4.1,
    Equations (11) and (12), normalized to rho c = 400 N s/m3.

        LW = Lp + 10 lg(A/A0) + 4.34 A/S + 10 lg(1 + S c / (8 V f))
             - 25 lg(427/400 sqrt(273/(273 + theta)) B/B0) - 6 dB

    with A = 55.26 V / (c T), c = 20.05 sqrt(273 + theta) m/s, A0 = 1 m2 and
    B0 = 1.013e5 Pa. The term 4.34 A/S accounts for air absorption in the
    room and 10 lg(1 + S c / (8 V f)) is the Waterhouse correction for the
    energy near the room boundaries.

    Parameters (default), (type):
    -----------------------------

        * Lp (), (array-like):
            Average sound pressure level in the room by band, in dB, after
            the background correction (8.3), e.g. the data of Lp_ST;

        * T (), (array-like):
            Reverberation time by band with the source installed, in s;

        * V (), (float):
            Room volume, in m3;

        * S (), (float):
            Total surface area of the room, in m2;

        * f (), (array-like):
            Midband frequencies of the bands, in Hz;

        * temperature (), (float):
            Air temperature in the room, in degC;

        * pressure (), (float):
            Atmospheric pressure, in Pa.

    Returns:
    ---------

        numpy.ndarray with LW by band, in dB.

    """
    c = 20.05 * np.sqrt(273 + temperature)
    A = 55.26 * V / (c * np.asarray(T, dtype=float))
    impedance = 427 / 400 * np.sqrt(273 / (273 + temperature)) \
        * pressure / 1.013e5
    return np.asarray(Lp, dtype=float) + 10 * np.log10(A) + 4.34 * A / S \
        + 10 * np.log10(1 + S * c / (8 * V * np.asarray(f, dtype=float))) \
        - 25 * np.log10(impedance) - 6


def sound_power_comparison(Lp, Lp_ref, LW_ref):
    """
    Sound power level by the comparison method of ISO 3741:1999, 8.4.2,
    Equation (13): LW = LWr + (Lp - Lpr).

    Parameters (default), (type):
    -----------------------------

        * Lp (), (array-like):
            Average sound pressure level in the room by band with the source
            under test, in dB (8.3);

        * Lp_ref (), (array-like):
            Average sound pressure level in the room by band with the
            reference sound source, in dB (8.4.2.2);

        * LW_ref (), (array-like):
            Sound power level of the reference sound source by band, in dB,
            as calibrated for rho c = 400 N s/m3 and not corrected for the
            conditions of the test.

    Returns:
    ---------

        numpy.ndarray with LW by band, in dB.

    """
    return np.asarray(LW_ref, dtype=float) + np.asarray(Lp, dtype=float) \
        - np.asarray(Lp_ref, dtype=float)
