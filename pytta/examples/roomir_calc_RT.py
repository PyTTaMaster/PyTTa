#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 19:14:51 2020

@author: mtslazarin
"""

# %% Importando bibliotecas
import pytta
import os
import time
import numpy as np
import copy as cp
import gc

# %% Muda o current working directory do Python para a pasta onde este script
# se encontra
cwd = os.path.dirname(__file__)  # Pega a pasta de trabalho atual
os.chdir(cwd)

# %% Opções de pós processamento
analysisFileName = '11_05_20_TR_inSitu'
MPP = pytta.roomir.MeasurementPostProcess(nthOct=3,
                                          minFreq=50,
                                          maxFreq=10000)
IREndManualCut = 7
skipInCompensation = True  # Não aplicar compensação da cadeia de entrada
skipOutCompensation = True  # Não aplicar compensação da cadeia de saída
skipBypCalibration = True  # Não aplicar calibração bypass
skipIndCalibration = True  # Não aplicar calibração indireta
skipRegularization = False  # Não aplicar regularização
skipSave = True  # Não salvar RIs calculadas para este processamento
plots = False

# %% Carrega medição
MS_inSitu1, D_inSitu1 = pytta.roomir.med_load('med-final_inSitu-1')

# %% Calcula resposta impulsiva bypass (saída conectada à entrada da interface)
if not skipBypCalibration:
    getChCalib = D_inSitu1.get('channelcalibration', 'Mic1')
    ChCalibir = D_inSitu1.calculate_ir(getChCalib,
                                       calibrationTake=1,
                                       skipSave=False)
    for name, msdThng in ChCalibir.items():
        print(name)
    if plots:
        #p1 = [IR.plot_freq(ylim=[-10, 120]) for IR in msdThng.measuredSignals]
        p2 = msdThng.measuredSignals[0].plot_freq(xLim=None)
        p3 = msdThng.measuredSignals[0].plot_time_dB(xLim=None)

# %% Dicionário que armazenará 
TR_inSitu = {}

#%% Calcula measuredThings tipo roomir (RIs) p/ estimativa do TR em todas
# configurações fonte receptor
for source in [1, 2]:
    for receiver in [1, 2, 3]:
        gc.collect()
        SR = "S" + str(source) + "-R" + str(receiver)
        getRoomres = D_inSitu1.get('roomres', 'Mic1', SR)
        SR = SR.replace('-','')
        RoomirsInSitu1 = D_inSitu1.calculate_ir(getRoomres,
                           calibrationTake=1,
                           skipInCompensation=skipInCompensation,
                           skipOutCompensation=skipOutCompensation,
                           skipBypCalibration=skipBypCalibration,
                           skipIndCalibration=skipIndCalibration,
                           skipRegularization=skipRegularization,
                           skipSave=skipSave)
        
        # Plota respostas impulsivas de cada cfg fonte-receptor
        for name, msdThng in RoomirsInSitu1.items():
            print(name)
        if plots:
            #p1 = [IR.plot_freq(ylim=[-10, 120]) for IR in msdThng.measuredSignals]
            p2 = msdThng.measuredSignals[0].plot_freq(xLim=None)
            p3 = msdThng.measuredSignals[0].plot_time_dB(xLim=None)
            
        # Calcula TR
        newTR = MPP.TR(RoomirsInSitu1, 20, IREndManualCut)
    
        if plots:
            _ = newTR[SR][0].plot()
    
        TR_inSitu[SR] = newTR[SR]
    
# Salva resultados
TR_inSitu['dictName'] = 'TR_inSitu'
pytta.save(analysisFileName + ".hdf5", TR_inSitu)
