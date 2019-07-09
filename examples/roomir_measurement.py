#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:08:25 2019

@author: mtslazarin
"""

import pytta
from pytta import roomir as m
from pytta.classes import lju3ei1050

#%% Cria objeto para stream de dados com o LabJack U3 com o sensor 
# de temperatura e umidade EI1050

# tempHumid = lju3ei1050.main()
tempHumid = None  # Para testes com LabJack offline

#%% Carrega sinais de excitação
excitationSignals = {}
excitationSignals['varredura'] = pytta.generate.sweep(
        # Geração do sweep (também pode ser carregado projeto prévio)
        freqMin=20,
        freqMax=20000,
        fftDegree=19,
        startMargin=0.75,
        stopMargin=1.5,
        method='logarithmic',
        windowing='hann')
# Carregando sinal de música
excitationSignals['musica'] = pytta.read_wav(
        'audio/Piano Over the rainbow Mic2 SHORT_edited.wav')
# Carregando sinal de fala
excitationSignals['fala'] = pytta.read_wav(
        'audio/Voice Sabine Short_edited.WAV')

#%% Cria novo Setup de Medição
SM = m.newMeasurement(name='med-teste',  # Nome da medição
                      # Sintaxe : device = [<in>,<out>] ou <in/out>
                      # Utilize pytta.list_devices() para listar
                      # os dispositivos do seu computador.
                      # device=[0,1], # PC laza
                      device=4,  # Saffire Pro 40 laza
                      # device=[1,3], # PC Leo
                      # device=0, # Firebox laza
                      # device=[1,4], # PC laza
                      # Sinais de excitação
                      excitationSignals=excitationSignals,
                      samplingRate=44100,  # [Hz]
                      freqMin=20,  # [Hz]
                      freqMax=20000,  # [Hz]
                      # Dicionário com códigos e canais de saída associados
                      inChannels={'OE': (1, 'Orelha E'),
                                 'OD': (2, 'Orelha D'),
                                 'Mic1': (4, 'Mic 1'),
                                 'Mic2': (5, 'Mic 2'),
                                 'combinedChannels': [(1,2)]},
                      # Dicionário com códigos e canais de saída associados
                      outChannels={'S1': (1, 'Dodecaedro 1'),
                                  'S2': (2, 'Dodecaedro 2'),
                                  'S3': (3, 'Sistema da sala'),
                      averages=3,  # Número de médias por medição
                      # Número de fontes; dodecaedro e p.a. local
                      sourcesNumber=3,
                      # Número de receptores
                      receiversNumber=5,
                      # [s] tempo de gravação do ruído de fundo
                      noiseFloorTp=5,
                      # [s] tempo de gravação do sinal de calibração
                      calibrationTp=2)

#%% Cria instância de dados medidos
D = m.Data(SM)
# D.dummyFill() # Preenche instância de dados com dummy signals

#%% Mostra status da instância de dados medidos
D.getStatus()

#%% Cria nova tomada de medição para uma nova configuração fonte receptor
measureTake = m.measureTake(SM,
                            kind='newpoint',
                            # Status do canal:
                            # True para Ativado e False para Desativado
                            channelStatus=[True,  # canal 1
                                           True,  # canal 2
                                           True,  # canal 3
                                           True],  # canal 4
                            # Configuração fonte receptor
                            # Obs. 1: manter itens da lista para
                            #         canais Desativados;
                            # Obs. 2: canais combinados devem ter a mesma cfg.
                            receivers=['R1',  # canal 1
                                      'R1',  # canal 2
                                      'R2',  # canal 3
                                      'R3'],  # canal 4
                            # Código de fonte a ser utilizado.
                            # Para fins de seleção dos canais de saída
                            # source='S1',
                            source='S2',
                            # source='S3',
                            # escolhe sinal de excitacão
                            # disponível no Setup de Medição
                            excitation='varredura',
                            # excitation='fala',
                            # excitation='musica',
                            # Passa objeto de comunicação
                            # com o LabJack U3 + EI1050 probe
                            tempHumid=tempHumid)
#%% Cria nova tomada de medição do ruído de fundo
measureTake = m.measureTake(SM,
                            kind='noisefloor',
                            # Status do canal:
                            # True para Ativado e False para Desativado
                            channelStatus=[True,  # canal 1
                                           True,  # canal 2
                                           True,  # canal 3
                                           True],  # canal 4
                            # Configuração fonte receptor
                            # Obs. 1: manter itens da lista para
                            #         canais Desativados;
                            # Obs. 2: canais combinados devem ter a mesma cfg;
                            # Obs. 3: para kind = 'noisefloor' não há fonte.
                            receivers=['R1',  # canal 1
                                      'R1',  # canal 2
                                      'R2',  # canal 3
                                      'R3'],  # canal 4
                            # Passa objeto de comunicação
                            # com o LabJack U3 + EI1050 probe
                            tempHumid=tempHumid)
#%% Cria nova tomada de medição para calibração
measureTake = m.measureTake(SM,
                            kind='calibration',
                            # Status do canal:
                            # True para Ativado e False para Desativado
                            # Obs. 1: para kind = 'calibration' os canais devem
                            #         ser calibrados individualmente.
                            channelStatus=[True,  # canal 1
                                           True,  # canal 2
                                           True,  # canal 3
                                           True],  # canal 4
                            tempHumid = tempHumid) # passa objeto de comunicação com LabJack U3 + EI1050
#%% Nova tomada de medição
measureTake.run()

#%% Salva tomada de medição no objeto de dados D e no disco
measureTake.save(D)

#%% Carrega dados medidos e setup de medição do arquivo
SM, D = m.load('med-teste')
