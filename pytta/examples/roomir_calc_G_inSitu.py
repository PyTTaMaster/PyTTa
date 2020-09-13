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
analysisFileName = '11_05_20_G_inSitu-ganho1_sem-nada'
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

# %% Loading reverberation chamber G terms
#a = pytta.load('G_revCh.hdf5')
pathRevCh = '/Users/mtslazarin/.../Dados medidos/13_12_19_med-final_revCh03/'
a = pytta.load(pathRevCh + '11_05_20_revCh_sem-nada.hdf5')
for key, value in a.items():
    exec(key + ' = value')
    exec(key + ".creation_name = '" + key + "'")
V_revCh = 207

# %%
MS_inSitu1, D_inSitu1 = pytta.roomir.med_load('med-final_inSitu-1')

# %% Calcula channelcalibir
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
    
#%% Calcula sourcerecalibir p/ cálculo do Lps
getSourcerecalib = D_inSitu1.get('sourcerecalibration', 'Mic1')
Recalibirs1 = D_inSitu1.calculate_ir(getSourcerecalib,
                                       calibrationTake=1,
                                       skipInCompensation=skipInCompensation,
                                       skipOutCompensation=skipOutCompensation,
                                       skipBypCalibration=skipBypCalibration,
                                       skipIndCalibration=skipIndCalibration,
                                       skipRegularization=skipRegularization,
                                       skipSave=skipSave)
for name, msdThng in Recalibirs1.items():
    print(name)
if plots:
    #p1 = [IR.plot_freq(ylim=[-10, 120]) for IR in msdThng.measuredSignals]
    p2 = msdThng.measuredSignals[0].plot_freq(xLim=None)
    p3 = msdThng.measuredSignals[0].plot_time_dB(xLim=None)
    
# %% Lps_inSitu
removerMediaIdx = [0,2,3]
for idx in sorted(removerMediaIdx, reverse=True):
    for name, msdThng in Recalibirs1.items():
        msdThng.measuredSignals.pop(idx)
Lps_inSitu1 = MPP.G_Lps(Recalibirs1)
Lps_inSitu1 = Lps_inSitu1[list(Lps_inSitu1.keys())[0]]
Lps_inSitu1.creation_name = 'Lps_inSitu1'
if plots:
    _ = Lps_inSitu1.plot(yLim=[0,46])
    _ = Lps_revCh.plot(yLim=[0,46])

# %%
G1 = {}

#%% Calcula roomirs p/ cálculo do Lpe
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
                           skipIndCalibration=skipIndCalibration
                           skipRegularization=skipRegularization,
                           skipSave=skipSave)
        for name, msdThng in RoomirsInSitu1.items():
            print(name)
        if plots:
            #p1 = [IR.plot_freq(ylim=[-10, 120]) for IR in msdThng.measuredSignals]
            p2 = msdThng.measuredSignals[0].plot_freq(xLim=None)
            p3 = msdThng.measuredSignals[0].plot_time_dB(xLim=None)
            
        # Lpe médio para cada posição
        # Lpe_inSitu = {'S1R1': [Analysis_avg1, Analysis_avg2, ..., Analysis_avgn]}            
        Lpe_inSitu1 = MPP.G_Lpe_inSitu(RoomirsInSitu1, IREndManualCut=IREndManualCut)
        
        if plots:
            _ = Lpe_inSitu1[SR][0].plot()
        
        # G in all positions loaded in Lpe_inSitu1
        newG = MPP.G(Lpe_inSitu1, Lpe_revCh, V_revCh, T_revCh, Lps_revCh, Lps_inSitu1)
        _ = G1[SR].plot(yLim=None, title='G in situ ganho 1',
                        xLabel='Bandas de frequência [Hz]',
                        yLabel='Fator de força [dB]')
        
        G1[SR] = newG[SR]
        # Save
        pytta.save(analysisFileName + "_" + SR + ".hdf5", G1, Lpe_inSitu1,
                   Lps_inSitu1)

# %%
if len(G1AllPos.keys()) == 6:
    pytta.save(analysisFileName + ".hdf5", G1AllPos, Lpe_inSitu1,
                       Lps_inSitu1)
        
# %%
G1 = {'dictName':'G1'}
Lpe_inSitu1 = {'dictName':'Lpe_inSitu1'}

#%%
for source in [1, 2]:
    for receiver in [1, 2, 3]:
        SR = "S" + str(source) + "R" + str(receiver)
        a = pytta.load(analysisFileName + "_" + SR + ".hdf5")
        G1[SR] = a['G1'][SR]
        Lps_inSitu1 = a['Lps_inSitu1']
        Lpe_inSitu1[SR] = a['Lpe_inSitu1'][SR]

pytta.save(analysisFileName + ".hdf5", G1, Lpe_inSitu1, Lps_inSitu1)
    