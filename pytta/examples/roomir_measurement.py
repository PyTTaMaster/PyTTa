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
from pytta.classes import lju3ei1050 # Comunicação c/ o LabJack U3 + EI1050
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

# %% Muda o current working directory do Python para a pasta onde este script
# se encontra
cwd = os.path.dirname(__file__) # Pega a pasta de trabalho atual
os.chdir(cwd)

# %%
# Cria objeto para stream de dados com o LabJack U3 com o sensor
# de temperatura e umidade EI1050
# tempHumid = lju3ei1050.main()
tempHumid = None  # Para testes com LabJack offline

# %% Caso já tenha uma medição em curso, carregue o MeasurementSetup e 
# MeasurementData
MS, D = rmr.med_load('med-teste')

# %% Carrega sinais de excitação e cria dicionário para o setup da medição
excitationSignals = {}
excitationSignals['varredura'] = pytta.generate.sweep(
        # Geração do sweep (também pode ser carregado projeto prévio)
        freqMin=20,
        freqMax=20000,
        fftDegree=19,
        startMargin=0.05,
        stopMargin=3,
        method='logarithmic',
        windowing='hann',
        samplingRate=48000)
# Carregando sinal de música
excitationSignals['musica'] = pytta.read_wav(
        'audio/Piano Over the rainbow Mic2 SHORT_edited.wav')
# Carregando sinal de fala
excitationSignals['fala'] = pytta.read_wav(
        'audio/Voice Sabine Short_edited.WAV')

# %% Carrega sensibilidade do microfone para compensação na cadeia de entrada
loadtxt = np.loadtxt(fname='14900_no_header.txt')
mSensFreq = loadtxt[:,0]
mSensdBMag = loadtxt[:,1]
# plt.semilogx(mSensFreq, mSensdBMag)

# %% Carrega resposta da fonte para compensação na cadeia de saida
matLoad = io.loadmat('hcorrDodecCTISM2m_Brum.mat')
ir = pytta.SignalObj(matLoad['hcorrDodecCTISM2m'],'time',44100)
sSensFreq = ir.freqVector
sourceSens = 1/np.abs(ir.freqSignal)[:,0]
normIdx = np.where(sSensFreq > 1000)[0][0]
sourceSens = sourceSens/sourceSens[normIdx]
sSensdBMag = 20*np.log10(sourceSens)
# plt.semilogx(sSensFreq, sSensdBMag)

# %% Cria novo setup de medição e inicializa objeto de dados, que gerencia o
# MeasurementSetup e os dados da medição em disco
MS = rmr.MeasurementSetup(name='med-teste',  # Nome da medição
                          samplingRate=48000,  # [Hz]
                          # Sintaxe : device = [<in>,<out>] ou <in/out>
                          # Utilize pytta.list_devices() para listar
                          # os dispositivos do seu computador.
                          #   device=[0, 1],  # PC laza
                          device=4,  # Saffire Pro 40 laza
                          # device=[1, 3], # PC Leo
                        #   device=0,  # Firebox laza
                          # [s] tempo de gravação do ruído de fundo
                          noiseFloorTp=5,
                          # [s] tempo de gravação do sinal de calibração
                          calibrationTp=2,
                          # Sinais de excitação
                          excitationSignals=excitationSignals,
                          averages=2,  # Número de médias por medição
                          pause4Avg=False,  # Pausa entre as médias
                          freqMin=20,  # [Hz]
                          freqMax=20000,  # [Hz]
                          # Dicionário com códigos e canais de saída associados
                          inChannels={'OE': (4, 'Orelha E'),
                                      'OD': (3, 'Orelha D'),
                                      'Mic1': (1, 'Mic 1'),
                                      'Mic2': (2, 'Mic 2'),
                                      'groups': {'HATS': (4, 3)}},
                          # Dicionário com códigos dos canais e compensações
                          # associadas à cadeia de entrada
                          inCompensations={'Mic1': (mSensFreq, mSensdBMag)},
                          # Dicionário com códigos e canais de saída associados
                          outChannels={'O1': (1, 'Dodecaedro 1'),
                                       'O2': (2, 'Dodecaedro 2'),
                                       'O3': (4, 'Sistema da sala')},
                          # Dicionário com códigos dos canais e compensações
                          # associadas à cadeia de saída
                          outCompensations={'O2': (sSensFreq, sSensdBMag)})
                        #   outCompensations={})
D = rmr.MeasurementData(MS)

# %% Cria nova tomada de medição
takeMeasure = rmr.TakeMeasure(MS=MS,
                              # Passa objeto de comunicação
                              # com o LabJack U3 + EI1050 probe
                              tempHumid=tempHumid,
                              kind='roomres',
                              # Lista com códigos de canal individual ou
                              # códigos de grupo
                        #       inChSel=['HATS', 'Mic1'],
                              inChSel=['Mic1'],
                              # Configuração sala-fonte-receptor:
                              # Lista com as respectivas posições dos canais
                              # individuais ou grupos de canais de entrada
                              # selecionados
                        #       receiversPos=['R1', 'R2', 'R1'],
                              receiversPos=['R1'],
                              # Escolha do sinal de excitacão
                              # disponível no Setup de Medição
                              excitation='varredura',
                              # excitation='fala',
                              # excitation='musica',
                              # Código do canal de saída a ser utilizado.
                              outChSel='O2',
                              # Ganho na saída
                              outputAmplification=-3, # [dB]
                              # Configuração sala-fonte-receptor
                              sourcePos='S1')

# %% Cria nova tomada de medição do ruído de fundo
takeMeasure = rmr.TakeMeasure(MS=MS,
                              # Passa objeto de comunicação
                              # com o LabJack U3 + EI1050 probe
                              tempHumid=tempHumid,
                              kind='noisefloor',
                              # Lista com códigos de canal individual ou
                              # códigos de grupo
                        #       inChSel=['Mic1', 'HATS'],
                              inChSel=['Mic1'],
                              # Configuração sala-receptor:
                              # Lista com as respectivas posições dos canais
                              # individuais ou grupos de canais de entrada
                              # selecionados
                        #       receiversPos=['R1', 'R2'])
                              receiversPos=['R1'])

# %% Cria nova tomada de medição para recalibração de fonte
takeMeasure = rmr.TakeMeasure(MS=MS,
                              # Passa objeto de comunicação
                              # com o LabJack U3 + EI1050 probe
                              tempHumid=tempHumid,
                              kind='sourcerecalibration',
                              # Lista com códigos de canal individual ou
                              # códigos de grupo
                              inChSel=['Mic1'],
                              # Escolha do sinal de excitacão
                              # disponível no Setup de Medição
                              excitation='varredura',
                              # Código do canal de saída a ser utilizado.
                              outChSel='O2',
                              # Ganho na saída
                              outputAmplification=-6) # [dB]

# %% Cria nova tomada de medição para calibração do microfone
takeMeasure = rmr.TakeMeasure(MS=MS,
                              # Passa objeto de comunicação
                              # com o LabJack U3 + EI1050 probe
                              tempHumid=tempHumid,
                              kind='miccalibration',
                              # Lista com códigos de canal individual ou
                              # códigos de grupo
                              inChSel=['Mic1'])

# %% Cria nova tomada de medição para calibração de canal
takeMeasure = rmr.TakeMeasure(MS=MS,
                              # Passa objeto de comunicação
                              # com o LabJack U3 + EI1050 probe
                              tempHumid=tempHumid,
                              kind='channelcalibration',
                              # Lista com códigos de canal individual ou
                              # códigos de grupo
                              inChSel=['Mic1'],
                              # Escolha do sinal de excitacão
                              # disponível no Setup de Medição
                              excitation='varredura',
                              # Código do canal de saída a ser utilizado.
                              outChSel='O2',
                              # Ganho na saída
                              outputAmplification=-5) # [dB]
# %% Inicia tomada de medição/aquisição de dados
takeMeasure.run()

# %% Salva tomada de medição no disco
D.save_take(takeMeasure)

# %% Carrega um dicionário com MeasuredThings de acordo com as tags fornecidas
# e faz algum processamento
a = D.get('channelcalibir', 'Mic1')
# msdThing = a['roomres_S1-R1_O1-Mic1_varredura_1']
msdThing = a['channelcalibir_O1-Mic1_varredura_1']
msdThing.measuredSignals[0].plot_time()
msdThing.measuredSignals[0].plot_freq()

# %% Calcula respostas ao sinal de excitação calibradas e salva em disco (vide 
# parâmetro skipSave)
a = D.get('roomres', 'Mic1')
b = D.calibrate_res(a, calibrationTake=1, skipSave=False)
for name, res in b.items():
        print(name)
        # res.measuredSignals[0].plot_time()
        res.measuredSignals[0].plot_freq()

# %% Calcula respostas impulsivas aplicando calibrações e salva em disco
a = D.get('roomres', 'Mic1')
b = D.calculate_ir(a,
                   calibrationTake=1,
                   skipInCompensation=False, # Ok
                   skipOutCompensation=False, # Ok
                   skipBypCalibration=False, # Ok
                   skipIndCalibration=False, # Ok
                   skipRegularization=False, # Ok
                   skipSave=False)
for name, IR in b.items():
        print(name)
        # IR.measuredSignals[0].plot_time()
        prot1 = IR.measuredSignals[0].plot_freq(xlim=[1, 24000], ylim=[60,100])
        # prot1 = IR.measuredSignals[0].plot_freq(xlim=[20, 20000], ylim=[20,96])
        # prot1 = IR.measuredSignals[0].plot_freq(xlim=None)
        # prot2 = IR.measuredSignals[0].plot_time(xlim=[0,0.004])
        prot2 = IR.measuredSignals[0].plot_time(xlim=None)

# %% Formas alternativas de carregar dados na memória

# %% Carrega MS e todas as MeasuredThings
a = rmr.h5_load(MS.name + '/MeasurementData.hdf5')

# %% Carrega sinais de excitação utilizados
a = rmr.h5_load(MS.name + '/MeasurementData.hdf5', skip=['MeasuredThing'])
loadedExcitationSignals = a['MeasurementSetup'].excitationSignals
loadedExcitationSignals['varredura'].plot_freq()

# %%
b['roomir_S1-R1_O1-Mic1_varredura_1'].measuredSignals[0].systemSignal.plot_time_dB()

# %%
