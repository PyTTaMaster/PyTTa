#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Room Parameters.

Demonstration of how to use the new RoomParameters class.

@author: João Vitor Gutkoski Paes.
"""
import pytta


# myMonoIR = pytta.load("SOME_SAVED_SIGNALOBJ_WITH_IR.hdf5")
myMonoIR = pytta.read_wav("RIS/scene9_RIR_LS1_MP1_Dodecahedron.wav")


room = pytta.RoomAnalysis(myMonoIR, nthOct=3, minFreq=50., maxFreq=16e3)


print()
print("Parameters from impulse response are:")
print("\n", room.parameters, "\n")
print("Access directly by RoomAnalysis().PNAME")
print("Or view it in a bar plot by RoomAnalysis().plot_PNAME")
print("where PNAME is the name of the desired parameter, as shown above.")
print()
print(f"{room.T20=}")


fig = room.plot_EDT()
fig.show()
