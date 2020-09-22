#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:08:25 2019

@author: mtslazarin
"""
# %% Initializating
import pytta
from pytta import roomir as rmr
import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

# %% Muda o current working directory do Python para a pasta onde este script
# se encontra
cwd = os.path.dirname(__file__)  # Pega a pasta de trabalho atual
os.chdir(cwd)

# %% Opções de pós processamento
analysisFileName = '11_05_20_revCh_sem-nada.hdf5'
V_revCh = 207
MPP = pytta.roomir.MeasurementPostProcess(nthOct=3,
                                          minFreq=100,
                                          maxFreq=10000)
IREndManualCut = 7
skipInCompensation = True
skipOutCompensation = True
skipBypCalibration = True
skipIndCalibration = True
skipRegularization = False
skipSave = True
plots = False

# %% MeasurementData
MSrevCh, DrevCh = rmr.med_load('med-final_revCh-Lpe')
analysisFileNameTR = '11_05_20_revCh-TR.hdf5'

# %% Calcula channelcalibir
getChCalib = DrevCh.get('channelcalibration', 'Mic1')
ChCalibir = DrevCh.calculate_ir(getChCalib,
                         calibrationTake=1,
                         skipSave=skipSave)
for name, msdThng in ChCalibir.items():
    print(name)
if plots:
    #p1 = [IR.plot_freq(ylim=[-10, 120]) for IR in msdThng.measuredSignals]
    p2 = msdThng.measuredSignals[0].plot_freq(xLim=None)
    p3 = msdThng.measuredSignals[0].plot_time_dB(xLim=None)

#%% Calcula sourcerecalibir
getSourcerecalib = DrevCh.get('sourcerecalibration', 'Mic1')
Recalibirs = DrevCh.calculate_ir(getSourcerecalib,
                                       calibrationTake=1,
                                       skipInCompensation=skipInCompensation,
                                       skipOutCompensation=skipOutCompensation,
                                       skipBypCalibration=skipBypCalibration,
                                       skipIndCalibration=skipIndCalibration,
                                       skipRegularization=skipRegularization,
                                       skipSave=skipSave)
for name, msdThng in Recalibirs.items():
    print(name)
if plots:
    #p1 = [IR.plot_freq(ylim=[-10, 120]) for IR in msdThng.measuredSignals]
    p2 = msdThng.measuredSignals[0].plot_freq(xLim=None)
    p3 = msdThng.measuredSignals[0].plot_time_dB(xLim=None)

#%% Calcula roomirs
getRoomres = DrevCh.get('roomres', 'Mic1')
RoomirsLpe = DrevCh.calculate_ir(getRoomres,
                   calibrationTake=1,
                   skipInCompensation=skipInCompensation,
                   skipOutCompensation=skipOutCompensation,
                   skipBypCalibration=skipBypCalibration,
                   skipIndCalibration=skipIndCalibration,
                   skipRegularization=skipRegularization,
                   skipSave=skipSave)
for name, msdThng in RoomirsLpe.items():
    print(name)
if plots:
    #p1 = [IR.plot_freq(ylim=[-10, 120]) for IR in msdThng.measuredSignals]
    p2 = msdThng.measuredSignals[0].plot_freq(xLim=None)
    p3 = msdThng.measuredSignals[0].plot_time_dB(xLim=None)

# %% Calcula Lps_revCh
Lps_revCh =  MPP.G_Lps(Recalibirs)
Lps_revCh = Lps_revCh[list(Lps_revCh.keys())[0]]
Lps_revCh.creation_name = 'Lps_revCh'
if plots:
    _ = Recalibirs[list(Recalibirs.keys())[0]].measuredSignals[0].plot_time_dB()
    _ = Lps_revCh.plot()
    # _ = Lps_revCh.plot(yLim=[0,42])

# %% Lpe in all positions
Lpe_revCh = MPP.G_Lpe_revCh(RoomirsLpe, IREndManualCut=IREndManualCut)
if plots:
    _ = roomirsRevCh[list(roomirsRevCh.keys())[0]].measuredSignals[0].plot_time_dB()
    _ = Lpe_revCh.plot(errorLabel=None)
    # _ = Lpe_revCh.plot(yLim=[24,39])

# %% T_revCh
a = pytta.load(analysisFileNameTR)
for key, value in a.items():
    exec(key + ' = value')
    exec(key + ".creation_name = '" + key + "'")
if plots:
    _ = T_revCh.plot()

# %%
pytta.save(analysisFileName, Lps_revCh, Lpe_revCh, T_revCh)

# %%
a = pytta.load(analysisFileName)
for key, value in a.items():
    exec(key + 'ld = value')
    exec(key + "ld.creation_name = '" + key + "'")
V_revCh = 207