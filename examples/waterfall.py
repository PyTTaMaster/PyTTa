"""
@authors: rinaldipp
"""

import pytta
from pytta._plot import waterfall
import os

if __name__ == "__main__":

    path = 'RIS' + os.sep
    name = 'scene9_RIR_LS1_MP1_Dodecahedron'
    wav = '.wav'

    myIRsignal = pytta.read_wav(path + name + wav)

    figs = waterfall(myIRsignal,
                     fmin=100,
                     fmax=4000,
                     tmax=2,
                     delta=60,
                     step=2**11,
                     n=2**11,
                     freq_tick=None,
                     time_tick=None,
                     mag_tick=None,
                     show=True,
                     fill_value='pmin',
                     fill_below=True,
                     figRatio=[1, 1, 1],
                     camera=[2, 1, 2],
                     saveFig='example',
                     )
