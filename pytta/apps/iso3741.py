"""
PyTTa ISO3741 Analysis:
----------------------

    This module does some calculations compliant to ISO 3471 in order to obtain
    the acoustic power of a source.

    For now only the calculation of the mean time-average sound pressure level
    among the microphone positions is implemented. In the future we hope the community
    keep developing this module in order to provide all the calculation needed
    for measurements acordding to this standard.

"""

from pytta import SignalObj, OctFilter, Analysis
import numpy as np
from pytta.classes.filter import fractional_octave_frequencies as FOF

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

def Lp_ST(sigObjList, nthOct, minFreq, maxFreq):
    """
    Calculate from the provided list of one channel SignalObjs the mean 
    one-third-octave band time-averaged sound pressure level in the test room
    with the noise source under test in operation, Lp(ST).

    Correction for background noise not implemented because of the UFSM
    reverberation chamber's good isolation.

    """
    Leqs = []

    for idx, sigObj in enumerate(sigObjList):
        firstChNum = sigObj.channels.mapping[0]

        if not sigObj.channels[firstChNum].calibCheck:
            raise ValueError("SignalObj {} must be calibrated.".format(idx+1))

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
    Lp_ST.unit = 'dB'
    return Lp_ST