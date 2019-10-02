#!/usr/bin/env python3
# -*-coding = utf-8 -*-
# %%
import pytta
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit

# %% Loading IR
path = pytta.__path__[0] + '/examples/RIS/'
file = 'scene9_RIR_LS1_MP1_Dodecahedron.wav'
myarr = pytta.read_wav(path+file)

#path = '/home/joaovitor/repositorios/pytta/pytta/examples/RIS/'
#myload = io.loadmat(path+'RI_mono_2.mat')['MonoRIS2_time']
#myarr = SignalObj(myload, 'time', 44100)

# %% Analyse
an = pytta.rooms.analyse(myarr, 'RT', 20, nthOct=3, minFreq=60, maxFreq=20000,
                        plotLundebyResults=True)
an.plot()

# %% Ita result
an_ita = pytta.Analysis(anType='RT',nthOct=3,minBand=60,maxBand=20000,
                        data=[2.0226, 1.7139, 1.4615,
                        1.7127, 1.0890, 1.5395, 1.2965, 1.9011, 1.9835, 2.1028,
                        2.1225, 1.9030, 1.9064, 2.0137, 1.8834, 1.6736, 1.5220,
                        1.5677, 1.6691, 1.4698, 1.2754, 0.9378, 0.6863, 0.4889,
                        0.3776, 0.3113])
an_ita.plot()

#%%
dif = an - an_ita
dif.plot()
# %% Processing duration time
# setup = """from __main__ import myout"""

# src = """myedc = [edc for edc in myout]"""

# print(timeit(stmt=src, setup=setup, number=1, globals=globals()))

# setup = """from __main__ import myrt"""

# src = """myres = [rt for rt in myrt]"""

# print(timeit(stmt=src, setup=setup, number=1, globals=globals()))