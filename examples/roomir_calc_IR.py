#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 19:58:51 2020

@author: mtslazarin
"""

# %% Importando bibliotecas
import pytta
import os

# %% Muda o current working directory do Python para a pasta onde este script
# se encontra
cwd = os.path.dirname(__file__)  # Pega a pasta de trabalho atual
os.chdir(cwd)

# %% Carrega a medição
MS, D = pytta.roomir.med_load('med-final_inSitu-1')

# %% Carrega um dicionário com MeasuredThings de acordo com as tags fornecidas
# e faz algum processamento. Tags podem ser qualquer dado presente na 
# identificação de uma MeasuredThing (coisa medida)

a = D.get('channelcalibir', 'Mic1')
msdThing = a['channelcalibir_O1-Mic1_varredura_1']

# a = D.get('roomres', 'Mic1')
# msdThing = a['roomres_S1-R1_O1-Mic1_varredura_1']

msdThing.measuredSignals[0].plot_time()
msdThing.measuredSignals[0].plot_freq()
# %% Calcula respostas impulsivas aplicando calibrações e salva em disco.

# Calcula-se MeasuredThings do tipo roomirs
a = D.get('roomres', 'Mic1')  

# Calcula-se MeasuredThings do tipo channelcalibir
# a = D.get('channelcalibration', 'Mic1')

b = D.calculate_ir(a,
                   calibrationTake=1,
                   skipInCompensation=False, # Ok
                   skipOutCompensation=False, # Ok
                   skipBypCalibration=False, # Ok
                   skipIndCalibration=False, # Ok
                   skipRegularization=False, # Ok
                   IREndManualCut=None,
                   IRStartManualCut=None,
                   skipSave=False)
for name, IR in b.items():
        print(name)
        # IR.measuredSignals[0].plot_time()
        # prot1 = IR.measuredSignals[0].plot_freq(xlim=[1, 24000], ylim=[60,100])
        prot1 = IR.measuredSignals[0].plot_freq(xlim=[1, 24000], ylim=[0,85])
        # prot1 = IR.measuredSignals[0].plot_freq(xlim=[20, 20000], ylim=[20,96])
        # prot1 = IR.measuredSignals[0].plot_freq(xlim=None)
        prot2 = IR.measuredSignals[0].plot_time_dB(xlim=None)
        # prot2 = IR.measuredSignals[0].plot_time(xlim=[-0.01, 0.3])

# %% Calcula respostas ao sinal de excitação calibradas e salva em disco
a = D.get('roomres', 'Mic1')
b = D.calibrate_res(a,
                    calibrationTake=1,
                    skipInCompensation=True,
                    skipSave=False)
for name, res in b.items():
        print(name)
        # res.measuredSignals[0].plot_time()
        res.measuredSignals[0].plot_freq(ylim=[-13,50])

# %% Formas alternativas de carregar dados na memória.
# Use por sua conta e risco

# %% Carrega objeto MeasurementSetup e todas as MeasuredThings

tudo = pytta.roomir._h5_load(MS.name + '/MeasurementData.hdf5')

# %% Carrega sinais de excitação utilizados

a = pytta.roomir._h5_load(MS.name + '/MeasurementData.hdf5', skip=['MeasuredThing'])
loadedExcitationSignals = a['MeasurementSetup'].excitationSignals
loadedExcitationSignals['varredura'].plot_freq()
