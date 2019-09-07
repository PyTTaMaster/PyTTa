#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:35:05 2019

@author: mtslazarin
"""

# import pytta
from pytta.classes._base import ChannelObj, ChannelsList
from pytta import generate

# Dict with the measurementKinds
measurementKinds = {'roomir': 'PlayRecMeasure',
                    'noisefloor': 'RecMeasure',
                    'miccalibration': 'RecMeasure',
                    'sourcerecalibration': 'PlayRecMeasure'}


class MeasurementChList(ChannelsList):

    # Magic methods

    def __init__(self, kind, groups={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = kind
        self.groups = groups

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                # MeasurementChList properties
                f'kind={self.kind!r}, '
                f'groups={self.groups!r}, '
                # ChannelsList properties
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
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, newComb):
        if not isinstance(newComb, dict):
            raise TypeError('groups must be a dict with array name ' +
                            'as key and channel numbers in a tuple as value.')
        for arrayName, group in newComb.items():
            if not isinstance(group, tuple):
                raise TypeError('Groups of channels inside the ' +
                                'groups dict must be contained by' +
                                ' a tuple.')
                for chNum in group:
                    if chNum not in MeasurementChList.mapping:
                        raise ValueError('Channel number ' + str(chNum) +
                                         ' isn\'t a valid ' + self.kind +
                                         'put channel.')
        self._groups = newComb

    # Methods

    def is_grouped(self, chRef):
        # Check if chRef is in any group
        if isinstance(chRef, str):
            if chRef in self.codes:
                nameOrCode = 'code'
            elif chRef in self.names:
                nameOrCode = 'name'
            else:
                raise ValueError("Channel name/code doesn't exist.")
            combChRefList = []
            for comb in self.groups.values():
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
            for comb in self.groups.values():
                for chNum in comb:
                    combChNumList.append(chNum)
            return chRef in combChNumList

    def get_group_membs(self, chNumUnderCheck, *args):
        # Return a list with channel numbers in ChNumUnderCheck's group
        if 'rest' in args:
            rest = 'rest'
        else:
            rest = 'entire'
        othersCombndChs = []
        for comb in self.groups.values():
            if chNumUnderCheck in comb:
                # Get other ChNums in group
                for chNum in comb:
                    if chNum != chNumUnderCheck:
                        othersCombndChs.append(chNum)
                    else:
                        if rest == 'entire':
                            othersCombndChs.append(chNum)
        return tuple(othersCombndChs)

    def get_array_name(self, chNum):
        # Get chNum's array name
        for arname, group in self.groups.items():
            if chNum in group:
                return arname
        return None

    def copy_groups(self, mChList):
        # Copy groups from mChList containing any identical channel to self
        groups = {}
        for chNum in self.mapping:
            groupMapping = mChList.get_group_membs(
                    chNum, 'rest')
            for chNum2 in groupMapping:
                # Getting groups information for reconstructd
                # inChannels
                if self[chNum] == mChList[chNum2]:
                    groups[mChList.get_array_name(chNum)] =\
                        mChList.get_group_membs(chNum)
        self.groups = groups


class MeasurementSetup(object):

    def __init__(self,
                 name,
                 samplingRate,
                 device,
                 excitationSignals,
                 freqMin,
                 freqMax,
                 inChannels,
                 outChannels,
                 averages,
                 noiseFloorTp,
                 calibrationTp):
        self.measurementKinds = measurementKinds
        self.name = name
        self.samplingRate = samplingRate
        self.device = device
        self.noiseFloorTp = noiseFloorTp
        self.calibrationTp = calibrationTp
        self.excitationSignals = excitationSignals
        self.averages = averages
        self.freqMin = freqMin
        self.freqMax = freqMax
        self.inChannels = MeasurementChList(kind='in')
        for chCode, chContents in inChannels.items():
            if chCode == 'groups':
                self.inChannels.groups = chContents
            else:
                self.inChannels.append(ChannelObj(num=chContents[0],
                                                  name=chContents[1],
                                                  code=chCode))
        self.outChannels = MeasurementChList(kind='out')
        for chCode, chContents in outChannels.items():
            self.outChannels.append(ChannelObj(num=chContents[0],
                                               name=chContents[1],
                                               code=chCode))


class Data(object):

    # Magic methods

    def __init__(self, MS):
        self.raw = {}  # Creates empty dict for raw data
        for medkind in MS.measurementKinds:
            # Creates empty lists for each measurement kind
            self.raw[medkind] = []

    # Properties

    # Methods

    def getStatus():
        pass


class TakeMeasure(object):

    # Magic methods

    def __init__(self,
                 MS,
                 tempHumid,
                 kind,
                 inChSel,
                 receiversPos=None,
                 excitation=None,
                 outChSel=None,
                 sourcePos=None):
        self.MS = MS
        self.tempHumid = tempHumid
        if self.tempHumid is not None:
            self.tempHumid.start()
        self.kind = kind
        self.inChSel = inChSel
        self.receiversPos = receiversPos
        self.excitation = excitation
        self.outChSel = outChSel
        self.sourcePos = sourcePos
        self.__cfg_channels()
        self.__cfg_measurement_object

    def __cfg_channels(self):
        # Check for disabled combined channels
        if self.kind not in ['miccalibration', 'sourcerecalibration']:
            # Look for grouped channels through the active channels
            for idx, status in enumerate(self.inChSel):
                chNumUnderCheck = self.MS.inChannels.mapping[idx]
                if status is True:
                    othersCombndChs = self.MS.inChannels.get_group_membs(
                            chNumUnderCheck, 'rest')
                    for chNum in othersCombndChs:
                        chStatusIndex = self.MS.inChannels.mapping.index(chNum)
                        if self.inChSel[chStatusIndex] is False:
                            raise ValueError('Grouped input channel ' +
                                             str(chNum) + ' must be enabled ' +
                                             'because its other group member' +
                                             ', channel ' +
                                             str(chNumUnderCheck) +
                                             ', is also enabled')
        # Check for multiple channels activated during calibration
        else:
            #
            #  TO DO
            #
            pass
        # Constructing the inChannels list for the current take
        self.inChannels = MeasurementChList(kind='in')
        for idx, chStatus in enumerate(self.inChSel):
            chNum = self.MS.inChannels.mapping[j]
            if chStatus:
                self.inChannels.append(self.MS.inChannels[chNum])
        # Getting groups information for reconstructd
        # inChannels MeasurementChList
        self.inChannels.copy_groups(self.MS.inChannels)
        # Setting the outChannel for the current take
        self.outChannel = MeasurementChList(kind='out')
        self.outChannel.append(self.MS.outChannels[self.outChSel])

    def __cfg_measurement_object(self):
        # For roomir measurement kind
        if self.kind == 'roomir':
            self.measurementObject = \
                generate.measurement('playrec',
                                     excitation=self.MS.
                                     excitationSignals[self.excitation],
                                     samplingRate=self.MS.samplingRate,
                                     freqMin=self.MS.freqMin,
                                     freqMax=self.MS.freqMax,
                                     device=self.MS.device,
                                     inChannel=self.inChannels.mapping,
                                     outChannel=self.outChannel.mapping,
                                     comment='roomir')
        # For miccalibration measurement kind
        if self.kind == 'calibration':
            if self.inChSel.count(True) != 1:
                raise ValueError('Only one channel per calibration take!')
            self.measurementObject = \
                generate.measurement('rec',
                                     lengthDomain='time',
                                     timeLength=self.MS.calibrationTp,
                                     samplingRate=self.MS.samplingRate,
                                     freqMin=self.MS.freqMin,
                                     freqMax=self.MS.freqMax,
                                     device=self.MS.device,
                                     inChannel=self.inChannels.mapping,
                                     comment='calibration')
        # For noisefloor measurement kind
        if self.kind == 'noisefloor':
            self.measurementObject = \
                generate.measurement('rec',
                                     lengthDomain='time',
                                     timeLength=self.MS.noiseFloorTp,
                                     samplingRate=self.MS.samplingRate,
                                     freqMin=self.MS.freqMin,
                                     freqMax=self.MS.freqMax,
                                     device=self.MS.evice,
                                     inChannel=self.inChannels,
                                     comment='noisefloor')
        # For sourcerecalibration measurement kind
        if self.kind == 'sourcerecalibration':
            self.measurementObject = \
                generate.measurement('playrec',
                                     excitation=self.MS.
                                     excitationSignals[self.excitation],
                                     samplingRate=self.MS.samplingRate,
                                     freqMin=self.MS.freqMin,
                                     freqMax=self.MS.freqMax,
                                     device=self.MS.device,
                                     inChannel=self.inChannels.mapping,
                                     outChannel=self.outChannel.mapping,
                                     comment='sourcerecalibration')

    @property
    def MS(self):
        return self._MS

    @MS.setter
    def MS(self, newMS):
        if not isinstance(newMS, MeasurementSetup):
            raise TypeError('Measurement setup must be a MeasurementSetup ' +
                            'object.')
        self._MS = newMS

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, newKind):
        if not isinstance(newKind, str):
            raise TypeError('Measurement take Kind must be a string')
        if newKind not in self.MS.measurementKinds:
            raise ValueError('Measurement take Kind doesn\'t ' +
                             'exist in RoomIR application.')
        self._kind = newKind
        return

    @property
    def inChSel(self):
        return self._inChSel

    @inChSel.setter
    def inChSel(self, newChSelection):
        if not isinstance(newChSelection, list):
            raise TypeError('inChSel must be a list of booleans ' +
                            'with same number of itens as '+self.MS.name +
                            '\'s inChannels.')
        if len(newChSelection) < len(self._MS.inChannels):
            raise ValueError('inChSel\' number of itens must be the ' +
                             'same as ' + self.MS.name + '\'s inChannels.')
        for item in newChSelection:
            if not isinstance(item, bool):
                raise TypeError('inChSel must be a list of booleans ' +
                                'with the same number of itens as ' +
                                self.MS.name + '\'s inChannels.')
        self._inChSel = newChSelection

    @property
    def outChSel(self):
        return self._outChSel

    @outChSel.setter
    def outChSel(self, newChSelection):
        if not isinstance(newChSelection, str):
            raise TypeError('outChSel must be a string with a valid output ' +
                            'channel code listed in '+self.MS.name +
                            '\'s outChannels.')
        if newChSelection not in self.MS.outChannels:
            raise TypeError('Invalid outChSel code or name. It must be a ' +
                            'valid ' + self.MS.name + '\'s output channel.')
        self._outChSel = newChSelection

    @property
    def sourcePos(self):
        return self._sourcePos

    @sourcePos.setter
    def sourcePos(self, newSource):
        if not isinstance(newSource, str):
            if newSource is None and self.kind in ['noisefloor',
                                                   'calibration']:
                self._sourcePos = None
                return
            else:
                raise TypeError('Source must be a string.')
#        if newSource not in self.MS.outChannels:
#            raise ValueError(newSource + ' doesn\'t exist in ' +
#                             self.MS.name + '\'s outChannels.')
        self._sourcePos = newSource

    @property
    def receiversPos(self):
        return self._receiversPos

    @receiversPos.setter
    def receiversPos(self, newReceivers):
        if not isinstance(newReceivers, list):
            if newReceivers is None and self.kind in ['noisefloor']:
                self._receiversPos = None
                return
            else:
                raise TypeError('Receivers must be a list of strings ' +
                                'with same number of transducers and itens ' +
                                ' in ' + self.MS.name + '\'s inChannels ' +
                                '(e.g. [\'R1\', \'R5\', \'R13\'])')
        if len(newReceivers) < len(self._MS.inChannels):
            raise ValueError('Receivers\' number of itens must be the ' +
                             'same as ' + self.MS.name + '\'s inChannels.')
        for item in newReceivers:
            if item.split('R')[0] != '':
                raise ValueError(item + 'isn\'t a receiver position. It ' +
                                 'must start with \'R\' succeeded by It\'s ' +
                                 'number (e.g. R1).')
            else:
                try:
                    receiverNumber = int(item.split('R')[1])
                except ValueError:
                    raise ValueError(item + 'isn\'t a receiver position ' +
                                     'code. It must start with \'R\' ' +
                                     'succeeded by It\'s number (e.g. R1).')
#                if receiverNumber > self.MS.receiversNumber:
#                    raise TypeError('Receiver number out of ' + self.MS.name +
#                                    '\'s receivers range.')
        self._receiversPos = newReceivers
        return

    @property
    def excitation(self):
        return self._excitation

    @excitation.setter
    def excitation(self, newExcitation):
        if not isinstance(newExcitation, str):
            if newExcitation is None and self.kind in ['noisefloor',
                                                       'calibration']:
                self._excitation = None
                return
            else:
                raise TypeError('Excitation signal\'s name must be a string.')
        if newExcitation not in self.MS.excitationSignals:
            raise ValueError('Excitation signal doesn\'t exist in ' +
                             self.MS.name + '\'s excitationSignals')
        self._excitation = newExcitation
        return

    def run(self):
        self.measuredTake = []
#        if self.kind == 'newpoint':
        for i in range(0, self.MS.averages):
            self.measuredTake.append(self.measurementObject.run())
            # Adquire do LabJack U3 + EI1050 a temperatura e
            # umidade relativa instantânea
            if self.tempHumid is not None:
                self.measuredTake[i].temp, self.measuredTake[i].RH = \
                    self.tempHumid.read()
            else:
                self.measuredTake[i].temp, self.measuredTake[i].RH = \
                    (None, None)

    def save(self, dataObj):
        # Desmembra o SignalObj measureTake de 4 canais em 3 SignalObj
        # referentes ao arranjo biauricular em uma posição e ao centro
        # da cabeça em duas outras posições
        # TO DO
        pass


class MeasuredThing(object):

    # Magic methods

    def __init__(self,
                 kind,
                 arrayName,
                 measuredSignals,
                 inChannels,
                 position=(None, None),
                 excitation=None,
                 outChannel=None):
        self.kind = kind
        self.arrayName = arrayName
        self.position = position
        self.excitation = excitation
        self.measuredSignals = measuredSignals
        self.inChannels = inChannels
        self.outChannel = outChannel


class Transducer(object):

    # Magic methods

    def __init__(self, brand, model, serial, IR):
        self.brand = brand
        self.model = model
        self.serial = serial
        self.IR = IR
