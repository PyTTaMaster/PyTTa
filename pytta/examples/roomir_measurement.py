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
        fftDegree=17,
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
SM = m.MeasurementSetup(name='med-teste',  # Nome da medição
                        samplingRate=44100,  # [Hz]
                        # Sintaxe : device = [<in>,<out>] ou <in/out>
                        # Utilize pytta.list_devices() para listar
                        # os dispositivos do seu computador.
#                        device=[0, 1],  # PC laza
                         device=4,  # Saffire Pro 40 laza
                        # device=[1, 3], # PC Leo
                        # device=0, # Firebox laza
                        # device=[1, 4], # PC laza
                        # [s] tempo de gravação do ruído de fundo
                        noiseFloorTp=5,
                        # [s] tempo de gravação do sinal de calibração
                        calibrationTp=2,
                        # Sinais de excitação
                        excitationSignals=excitationSignals,
                        averages=3,  # Número de médias por medição
                        freqMin=20,  # [Hz]
                        freqMax=20000,  # [Hz]
                        # Dicionário com códigos e canais de saída associados
                        inChannels={'OE': (1, 'Orelha E'),
                                    'OD': (2, 'Orelha D'),
                                    'Mic1': (4, 'Mic 1'),
                                    'Mic2': (5, 'Mic 2'),
                                    'groups': {'HATS':(1, 2)}},
                        # Dicionário com códigos e canais de saída associados
                        outChannels={'O1': (1, 'Dodecaedro 1'),
                                     'O2': (2, 'Dodecaedro 2'),
                                     'O3': (3, 'Sistema da sala')})

#%% Cria instância de dados medidos
#D = m.Data(SM)
# D.dummyFill() # Preenche instância de dados com dummy signals

#%% Mostra status da instância de dados medidos
#D.getStatus()

#%% Cria nova tomada de medição para uma nova configuração fonte receptor
takeMeasure = m.TakeMeasure(MS=SM,
                            # Passa objeto de comunicação
                            # com o LabJack U3 + EI1050 probe
                            tempHumid=tempHumid,
                            kind='roomir',
                            # Status do canal:
                            # True para Ativado e False para Desativado
                            inChSel=[True,  # canal 1
                                     True,  # canal 2
                                     False,  # canal 3
                                     True],  # canal 4
                            # Configuração sala-fonte-receptor
                            # Obs. 1: manter itens da lista para
                            #         canais Desativados;
                            # Obs. 2: canais combinados devem ter a mesma cfg.
                            receiversPos=['R1',  # canal 1
                                          'R1',  # canal 2
                                          'R2',  # canal 3
                                          'R3'],  # canal 4
                            # escolhe sinal de excitacão
                            # disponível no Setup de Medição
                            excitation='varredura',
                            # excitation='fala',
                            # excitation='musica',
                            # Código do canal de saída a ser utilizado.
                            outChSel='O1',
                            # Configuração sala-fonte-receptor
                            sourcePos='S1')
#%% Cria nova tomada de medição do ruído de fundo

#%% Cria nova tomada de medição para calibração

#%% Nova tomada de medição
takeMeasure.run()

#%% Salva tomada de medição no objeto de dados D e no disco
takeMeasure.save(D)

#%% Carrega dados medidos e setup de medição do arquivo
SM, D = m.load('med-teste')
