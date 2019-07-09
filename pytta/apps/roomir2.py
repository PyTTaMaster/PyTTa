#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:35:05 2019

@author: mtslazarin
"""

# import pytta
from pytta.classes._base import ChannelObj, ChannelsList

_takeKinds = {'newpoint': None,
              'noisefloor': None,
              'calibration': None}


class MeasurementSetup(object):

    def __init__(self,
                 name,
                 device,
                 excitationSignals,
                 samplingRate,
                 freqMin,
                 freqMax,
                 inChannels,
                 outChannels,
                 averages,
                 sourcesNumber,
                 receiversNumber,
                 noiseFloorTp,
                 calibrationTp):
        self.name = name
        self.device = device
        self.excitationSignals = excitationSignals
        self.samplingRate = samplingRate
        self.freqMin = freqMin
        self.freqMax = freqMax
        self.inChannels = MeasurementChList(kind='in')
        for chCode, chContents in inChannels.items():
            if chCode == 'combinedChannels':
                self.inChannels.combinedChannels = chContents
            else:
                self.inChannels.append(ChannelObj(num=chContents[0],
                                                  name=chContents[1],
                                                  code=chCode))
        self.outChannels = MeasurementChList(kind='out')
        for chCode, chContents in outChannels.items():
            self.outChannels.append(ChannelObj(num=chContents[0],
                                               name=chContents[1],
                                               code=chCode))
        self.averages = averages
        self.sourcesNumber = sourcesNumber
        self.receiversNumber = receiversNumber
        self.noiseFloorTp = noiseFloorTp
        self.calibrationTp = calibrationTp


class Source(object):

    def __init__(self, name, code, coordinates, orientation):
        self.name = name
        self.code = code


class Receiver(object):

    def __init__(self, name, code, coordinates, orientation):
        self.name = name
        self.code = code


class MeasurementChList(ChannelsList):

    # Magic methods

    def __init__(self, kind, combinedChannels=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = kind
        self.combinedChannels = combinedChannels

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'kind={self.kind!r}, '
                f'combinedChannels={self.combinedChannels!r}, '
                f'chList={self._channels!r})')

    # Properties

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, newKind):
        if newKind == 'in' or newKind == 'out':
            self._kind = newKind
        else:
            raise ValueError('Kind must be \'in\' or \'out\'')

    @property
    def combinedChannels(self):
        return self._combinedChannels

    @combinedChannels.setter
    def combinedChannels(self, newComb):
        if not isinstance(newComb, list):
            raise TypeError('combinedChannels must be a list with tuples '
                            + 'containing the combined channels')
        for group in newComb:
            if not isinstance(group, tuple):
                raise TypeError('Groups of channels inside the ' +
                                'combinedChannels list must be contained by' +
                                ' a tuple.')
                for chNum in group:
                    if chNum not in MeasurementChList.mapping:
                        raise ValueError('Channel number ' + str(chNum) +
                                         ' isn\'t a valid ' + self.kind +
                                         'put channel.')
        self._combinedChannels = newComb

    # Methods

    def isCombined(self, chRef):
        if isinstance(chRef, str):
            if chRef in self.codes:
                nameOrCode = 'code'
            elif chRef in self.names:
                nameOrCode = 'name'
            else:
                raise ValueError("Channel name/code doesn't exist.")
            combChRefList = []
            for comb in self.combinedChannels:
                for chNum in comb:
                    if nameOrCode == 'code':
                        combChRefList.append(self.channels[chNum].code)
                    elif nameOrCode == 'name':
                        combChRefList.append(self.channels[chNum].name)
            return chRef in combChRefList
        elif isinstance(chRef, int):
            if chRef not in self.mapping:
                raise ValueError("Channel number doesn't exist.")
            combChNumList = []
            for comb in self.combinedChannels:
                for chNum in comb:
                    combChNumList.append(chNum)
            return chRef in combChNumList


class Data(object):
    pass


class MeasurementIR(object):

    # Magic methods

    def __init__(self, IR, inChannel, outChannel):
        self.IR = IR
        self.inChannel = inChannel
        self.outChannel = outChannel


class RoomIRs(object):

    # Magic methods

    def __init__(self, sourceCode, receiverCode):
        self.MIRs = []
        self.sourceCode = sourceCode
        self.receiverCode = receiverCode

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'sourceCode={self.sourceCode!r}, '
                f'receiverCode={self.receiverCode!r})')

    # Methods

    def append(self, newMIR):
        if isinstance(newMIR, MeasurementIR):
            self.MIRs.append(newMIR)
        else:
            raise TypeError("RoomIRs can contain only MeasurementIRs.")


class NoiseFloor(object):

    # Magic methods

    def __init__(self, NF, inChannel):
        self.NF = NF
        self.inChannel = inChannel


class RoomNFs(object):

    # Magic methods

    def __init__(self, receiverCode):
        self.NFs = []
        self.receiverCode = receiverCode

    # Methods

    def append(self, newNF):
        if isinstance(newNF, NoiseFloor):
            self.NFs.append(newNF)
        else:
            raise TypeError("RoomNFs can contain only NoiseFloors.")


class Calibration(object):

    # Magic methods

    def __init__(self, calibSignal):
        self.calibSignal = calibSignal


class Transducer(object):

    # Magic methods

    def __init__(self, brand, model, serial, IR):
        self.brand = brand
        self.model = model
        self.serial = serial
        self.IR = IR
