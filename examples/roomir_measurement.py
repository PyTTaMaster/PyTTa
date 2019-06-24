#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:08:25 2019

@author: mtslazarin
"""

import pytta
from pytta import roomir as m
from pytta.classes import lju3ei1050
from scipy import io
import numpy as np

#%% Cria objeto para stream de dados com o LabJack U3 com o sensor de temperatura e umidade EI1050

#tempHumid = lju3ei1050.main()
tempHumid = None # Para testes com LabJack offline

#%% Carrega sinais de excitação
excitationSignals = {}
excitationSignals['varredura'] = pytta.generate.sweep(freqMin=20, # Geração do sweep (também pode ser carregado projeto prévio)
                            freqMax=20000,
                            fftDegree=19,
                            startMargin=0.75,
                            stopMargin=1.5,
                            method='logarithmic',
                            windowing='hann')
excitationSignals['musica'] = pytta.read_wav('audio/Piano Over the rainbow Mic2 SHORT_edited.wav') # Carregando sinal de música
excitationSignals['fala'] = pytta.read_wav('audio/Voice Sabine Short_edited.WAV') # Carregando sinal de fala

#%% Cria novo Setup de Medição
SM = m.newMeasurement(name = 'med-teste', # Nome da medição
#                      Sintaxe : device = [<entrada>,<saida>] ou <entrada/saida>
#                      Utilize pytta.list_devices() para listar os dispositivos do seu computador. 
#                     device = [0,1], # PC laza Seleciona dispositivo listado em pytta.list_devices()
                     device = 4, # Saffire Pro 40 laza Seleciona dispositivo listado em pytta.list_devices()
#                     device = [1,3], # PC Leo Seleciona dispositivo listado em pytta.list_devices()
#                     device = 0, # Firebox laza Seleciona dispositivo listado em pytta.list_devices()
#                     device = [1,4], # PC laza Seleciona dispositivo listado em pytta.list_devices()
                     excitationSignals=excitationSignals, # Sinais de excitação
                     samplingRate = 44100, # [Hz]
                     freqMin = 20, # [Hz]
                     freqMax = 20000, # [Hz]
                     inChannel = [1,2,4,5], # Canais de entrada
                     inChName = ['Orelha E','Orelha D','Mic 1','Mic 2'], # Lista com o nome dos canais 
                     outChannel = {'S1':([1],'Dodecaedro 1'),
                                   'S2':([2],'Dodecaedro 2'),
                                   'S3':([3],'Sistema da sala')}, # Dicionário com códigos e canais de saída associados
                     averages = 3, # Número de médias por medição
                     sourcesNumber = 3, # Número de fontes; dodecaedro e p.a. local
                     receiversNumber = 5, # Número de receptores
                     noiseFloorTp = 5, # [s] tempo de gravação do ruído de fundo
                     calibrationTp = 2) # [s] tempo de gravação do sinal de calibração

#%% Cria instância de dados medidos
D = m.Data(SM)
# D.dummyFill() # Preenche instância de dados com dummy signals

#%% Mostra status da instância de dados medidos
D.getStatus()

#%% Cria nova tomada de medição para uma nova configuração fonte receptor
measureTake = m.measureTake(SM,
                            kind = 'newpoint',
                            # Status do canal: True para Ativado e False para Desativado
                            channelStatus = [True, # canal 1
                                             True, # canal 2
                                             True, # canal 3
                                             True], # canal 4
                            # Configuração fonte receptor
                            # Obs. 1: manter itens da lista para canais Desativados
                            receiver = ['R1', # canal 1 (ATENÇÃO: canal 1 e 2 devem ter a mesma cfg.)
                                        'R1', # canal 2 (ATENÇÃO: canal 1 e 2 devem ter a mesma cfg.)
                                        'R2', # canal 3 
                                        'R3'], # canal 4
#                            source = 'S1', # código de fonte a ser utilizado. Para fins de seleção dos canais de saída
                            source = 'S2',
#                            source = 'S3',
                            excitation = 'varredura', # escolhe sinal de excitacão  disponível no Setup de Medição
#                            excitation = 'fala',
#                            excitation = 'musica',
                            tempHumid = tempHumid) # passa objeto de comunicação com LabJack U3 + EI1050
#%% Cria nova tomada de medição do ruído de fundo
measureTake = m.measureTake(SM,
                            kind = 'noisefloor',
                            # Status do canal: True para Ativado e False para Desativado
                            channelStatus = [True, # canal 1
                                             True, # canal 2
                                             True, # canal 3
                                             True], # canal 4
                            # Configuração fonte receptor
                            # Obs. 1: manter itens da lista para canais Desativados
                            # Obs. 2: para kind = 'noisefloor' não há fonte
                            receiver = ['R1', # canal 1 (ATENÇÃO: canal 1 e 2 devem ter a mesma cfg.)
                                        'R1', # canal 2 (ATENÇÃO: canal 1 e 2 devem ter a mesma cfg.)
                                        'R2', # canal 4
                                        'R3'], # canal 3 
                            tempHumid = tempHumid) # passa objeto de comunicação com LabJack U3 + EI1050
#%% Cria nova tomada de medição para calibração
measureTake = m.measureTake(SM,
                            kind = 'calibration',
                            # Status do canal: True para Ativado e False para Desativado
                            # Obs. 1: para kind = 'calibration' os canais devem ser calibrados individualmente
                            channelStatus = [True, # canal 1
                                             False, # canal 2
                                             False, # canal 3
                                             False], # canal 4
                            tempHumid = tempHumid) # passa objeto de comunicação com LabJack U3 + EI1050
#%% Nova tomada de medição
measureTake.run()

#%% Salva tomada de medição no objeto de dados D e no disco
measureTake.save(D)

#%% Carrega dados medidos e setup de medição do arquivo
SM, D = m.load('med-teste')