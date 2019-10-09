#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:08:25 2019

@author: mtslazarin
"""
# %% Initializating
import pytta
from pytta import roomir as rmr
from pytta.classes import lju3ei1050
# %%
path = 'pytta/examples/'

# %%
# Cria objeto para stream de dados com o LabJack U3 com o sensor
# de temperatura e umidade EI1050
# tempHumid = lju3ei1050.main()
tempHumid = None  # Para testes com LabJack offline

# %% Carrega sinais de excitação
excitationSignals = {}
excitationSignals['varredura'] = pytta.generate.sweep(
        # Geração do sweep (também pode ser carregado projeto prévio)
        freqMin=20,
        freqMax=20000,
        fftDegree=14,
        startMargin=0.1,
        stopMargin=0.1,
        method='logarithmic',
        windowing='hann')
# Carregando sinal de música
excitationSignals['musica'] = pytta.read_wav(
        path + 'audio/Piano Over the rainbow Mic2 SHORT_edited.wav')
# Carregando sinal de fala
excitationSignals['fala'] = pytta.read_wav(
        path + 'audio/Voice Sabine Short_edited.WAV')

# %% Cria novo Setup de Medição
MS = rmr.MeasurementSetup(name='med-teste',  # Nome da medição
                          samplingRate=44100,  # [Hz]
                          # Sintaxe : device = [<in>,<out>] ou <in/out>
                          # Utilize pytta.list_devices() para listar
                          # os dispositivos do seu computador.
                          #   device=[0, 1],  # PC laza
                            device=4,  # Saffire Pro 40 laza
                          # device=[1, 3], # PC Leo
                        #   device=0,  # Firebox laza
                          # device=[1, 4], # PC laza
                          # [s] tempo de gravação do ruído de fundo
                          noiseFloorTp=5,
                          # [s] tempo de gravação do sinal de calibração
                          calibrationTp=2,
                          # Sinais de excitação
                          excitationSignals=excitationSignals,
                          averages=3,  # Número de médias por medição
                          pause4Avg=False,  # Pausa entre as médias
                          freqMin=20,  # [Hz]
                          freqMax=20000,  # [Hz]
                          # Dicionário com códigos e canais de saída associados
                          inChannels={'OE': (1, 'Orelha E'),
                                      'OD': (2, 'Orelha D'),
                                      'Mic1': (4, 'Mic 1'),
                                      'Mic2': (3, 'Mic 2'),
                                      'groups': {'HATS': (1, 2)}},
                          # Dicionário com códigos e canais de saída associados
                          outChannels={'O1': (1, 'Dodecaedro 1'),
                                       'O2': (2, 'Dodecaedro 2'),
                                       'O3': (3, 'Sistema da sala')})

# %% Initializate Data object, wich manage the measurement files in dics
D = rmr.MeasurementData(MS)

# %% Mostra status da instância de dados medidos
# D.getStatus()

# %% Cria nova tomada de medição para uma nova configuração fonte receptor
takeMeasure = rmr.TakeMeasure(MS=MS,
                              # Passa objeto de comunicação
                              # com o LabJack U3 + EI1050 probe
                              tempHumid=tempHumid,
                              kind='roomir',
                              # Lista com códigos de canal individual ou
                              # códigos de grupo
                              inChSel=['HATS', 'Mic1'],
                              # Configuração sala-fonte-receptor:
                              # Lista com as respectivas posições dos canais
                              # individuais ou grupos de canais de entrada
                              # selecionados
                              receiversPos=['R1', 'R2', 'R1'],
                              # Escolha do sinal de excitacão
                              # disponível no Setup de Medição
                              excitation='varredura',
                              # excitation='fala',
                              # excitation='musica',
                              # Código do canal de saída a ser utilizado.
                              outChSel='O1',
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
                              inChSel=['Mic1', 'HATS'],
                              # Configuração sala-receptor:
                              # Lista com as respectivas posições dos canais
                              # individuais ou grupos de canais de entrada
                              # selecionados
                              receiversPos=['R1', 'R2'])

# %% Cria nova tomada de medição para calibração do microfone
takeMeasure = rmr.TakeMeasure(MS=MS,
                              # Passa objeto de comunicação
                              # com o LabJack U3 + EI1050 probe
                              tempHumid=tempHumid,
                              kind='miccalibration',
                              # Lista com códigos de canal individual ou
                              # códigos de grupo
                              inChSel=['Mic1'])

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
                              outChSel='O1')
# %% Acquire data
takeMeasure.run()

# %% Salva tomada de medição no objeto de dados D e no disco
D.save_take(takeMeasure)

# %% Carrega dados medidos e setup de medição do arquivo
MS, D = rmr.med_load('med-teste')

# %% Carrega coisas medidas de acordo com as tags fornecidas
D.get('Mic1')

# %%