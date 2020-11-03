"""
PyTTa ISO3741 Analysis:
----------------------

This module does some calculations compliant to ISO 3471 in order to obtain
the acoustic power of a source.

For now only the calculation of the mean time-average sound pressure level
among the microphone positions is implemented. In the future we hope the
community keep developing this module in order to provide all the
calculation needed for measurements acordding to this standard.

Available functions:

    >>> pytta.iso3741.Lp_ST()

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
