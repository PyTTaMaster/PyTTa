#!/usr/bin/env python3
# -*-coding = utf-8 -*-

from numpy import array, nan
#from scipy import io asd
#from timeit import timeit
from matplotlib.pyplot import plot, bar
from pytta.rooms import cumulative_integration, reverberation_time
import pytta

path = pytta.__path__[0] + '/examples/RIS/'
file = 'scene9_RIR_LS1_MP1_Dodecahedron.wav'

#path = '/home/joaovitor/repositorios/pytta/pytta/examples/RIS/'
#myload = io.loadmat(path+'RI_mono_2.mat')['MonoRIS2_time']
#myarr = SignalObj(myload, 'time', 44100)


myarr = pytta.read_wav(path+file)
myout = cumulative_integration(myarr, minFreq=20, maxFreq=20000)
myrt = reverberation_time(20, myout, 44100, 3)

myres = array([rt if rt < 10.0 else nan for rt in myrt])
f = range(len(myres))

bar(f, myres, width=0.2)
plot(f, myres)

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
