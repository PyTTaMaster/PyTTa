#!/usr/bin/env python3
# -*-coding = utf-8 -*-
# %%
import pytta

# %% Loading IR
path = pytta.__path__[0] + '/examples/RIS/'
file = 'scene9_RIR_LS1_MP1_Dodecahedron.wav'
myarr = pytta.read_wav(path+file)

#path = '/home/joaovitor/repositorios/pytta/pytta/examples/RIS/'
#myload = io.loadmat(path+'RI_mono_2.mat')['MonoRIS2_time']
#myarr = SignalObj(myload, 'time', 44100)

# %% Analyse
an = pytta.rooms.analyse(myarr, 'RT', 20, 3, minFreq=20, maxFreq=20000)

# %% Processing duration time
#setup = """from __main__ import myout"""
#
#src = """myedc = [edc for edc in myout]"""
#
#print(timeit(stmt=src, setup=setup, number=1, globals=globals()))

#setup = """from __main__ import myrt"""
#
#src = """myres = [rt for rt in myrt]"""
#
#print(timeit(stmt=src, setup=setup, number=1, globals=globals()))