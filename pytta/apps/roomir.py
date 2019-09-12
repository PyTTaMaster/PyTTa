#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:35:05 2019

@author: mtslazarin
"""

from pytta.classes._base import ChannelObj, ChannelsList
from pytta import generate, SignalObj, h5save, h5load
import time
import numpy as np
import h5py
from os import getcwd, listdir, mkdir
from os.path import isfile, join, exists
from shutil import rmtree

# Dict with the measurementKinds
# TO DO: add 'inchcalibration', 'outchcalibration'
measurementKinds = {'roomir': 'PlayRecMeasure',
                    'noisefloor': 'RecMeasure',
                    'miccalibration': 'RecMeasure',
                    'sourcerecalibration': 'PlayRecMeasure'}


class MeasurementChList(ChannelsList):

    # Magic methods

    def __init__(self, kind, groups={}, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initializate the ChannelsList
        # Rest of initialization
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
                        combChRefList.append(self[chNum].code)
                    elif nameOrCode == 'name':
                        combChRefList.append(self[chNum].name)
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

    def get_group_name(self, chNum):
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
                    groups[mChList.get_group_name(chNum)] =\
                        mChList.get_group_membs(chNum)
        self.groups = groups


class MeasurementSetup(object):

    # Magic methods

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
                 pause4Avg,
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
        self.pause4Avg = pause4Avg
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
        self.path = getcwd()+'/'+self.name+'/'
        # Save MeasurementSetup to disc or warn if already exists
        if not exists(self.path):
            mkdir(self.path)
        elif exists(self.path + 'MeasurementSetup.hdf5'):
            # raise FileExistsError('ATTENTION!  MeasurementSetup for the ' +
            #                       ' current measurement, ' + self.name +
            #                       ', already exists. Load it instead of '
            #                       'overwriting.')
            print('Deleting the existant measurement: ' + self.MS.name)
            rmtree(self.path)
            mkdir(self.path)
            self.__h5_init()
        else:
            # Creating the MeasurementSetup file
            h5save('MeasurementSetup.hdf5', self)

    def __repr__(self):
        # TO DO
        pass

    # Methods

    def h5save(self, h5group):
        # TO DO
        pass


class MeasurementData(object):
    """
    Class dedicated to manage in the hard drive the acquired data stored as
    MeasuredThing objects.

    This class don't need a h5save method, as it saves itself into disc by
    its nature.
    """

    # Magic methods

    def __init__(self, MS):
        # MeasurementSetup
        self.MS = MS
        self.path = self.MS.path
        # Save MeasurementData to disc or warn if already exists
        if not exists(self.path):
            mkdir(self.path)
        elif exists(self.path + 'MeasurementData.hdf5'):
            # raise FileExistsError('ATTENTION!  MeasurementData for the ' +
            #                       ' current measurement, ' + self.MS.name +
            #                       ', already exists. Load it instead of '
            #                       'overwriting.')
            print('Deleting the existant measurement: ' + self.MS.name)
            rmtree(self.path)
            mkdir(self.path)
            self.__h5_init()
        else:
            self.__h5_init()

    # Methods

    def __h5_init(self):
        """
        Method for initializating a brand new MeasurementData.hdf5 file
        """
        # Creating the MeasurementData file
        with h5py.File(self.path + 'MeasurementData.hdf5', 'w-') as f:
            # Saving the MeasurementSetup link
            f['MeasurementSetup'] = h5py.ExternalLink(self.path +
                                                      'MeasurementSetup' +
                                                      '.hdf5',
                                                      '/MeasurementSetup')
            for msKind in self.MS.measurementKinds:
                # Creating groups for each measurement kind
                f.create_group(msKind)

    def save_take(self, MeasureTakeObj):
        if not MeasureTakeObj.runCheck:
            raise ValueError('Can\'t save an unacquired MeasuredThing. First' +
                             'you need to run the measurement through ' +
                             'TakeMeasure.run().')
        if MeasureTakeObj.saveCheck:
            raise ValueError('Can\'t save the this measurement take because ' +
                             'It has already been saved.')
        # Iterate over measuredThings
        for arrayName, measuredThing in MeasureTakeObj.measuredThings.items():
            fileName = str(measuredThing)
            # Checking if any measurement with the same configs was take
            fileName = self.__number_the_file(fileName)
            # Saving the MeasuredThing to the disc
            measuredThing.creation_name = fileName
            h5save(self.path + fileName, measuredThing)
            # Update the MeasurementData.hdf5 file with the MeasuredThing link
            with h5py.File(self.path + 'MeasurementData.hdf5', 'r+') as f:
                msdThngH5Group = f[measuredThing.kind].create_group(fileName)
                msdThngH5Group = h5py.ExternalLink(self.path +
                                                   fileName + '.hdf5',
                                                   '/' + fileName)
        MeasureTakeObj.saveCheck = True
        return

    def __number_the_file(self, fileName):
        """
        Search in the measurement folder if exist other take with the same
        name and rename the current fileName with the a counter at the end.
        """
        lasttake = 0
        myfiles = [f for f in listdir(self.path) if
                   isfile(join(self.path, f))]
        for file in myfiles:
            if fileName in file:
                newlasttake = file.replace(fileName + '_', '')
                try:
                    newlasttake = int(newlasttake.replace('.hdf5', ''))
                except ValueError:
                    newlasttake = lasttake
                if newlasttake > lasttake:
                    lasttake = newlasttake
        # Adding the counter to the fileName
        fileName += '_' + str(lasttake+1)
        return fileName

    def get_status():
        # TO DO
        pass

    # Properties


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
        self.__cfg_measurement_object()
        self.runCheck = False
        self.saveCheck = False

    # Methods

    def __cfg_channels(self):
        # Check for disabled combined channels
        if self.kind not in ['miccalibration', 'sourcerecalibration']:
            # Look for grouped channels through the individual channels
            for idx, code in enumerate(self.inChSel):
                if code not in self.MS.inChannels.groups:
                    chNum = self.MS.inChannels[code].num
                    if self.MS.inChannels.is_grouped(code):
                            raise ValueError('Input channel number' +
                                             str(chNum) + ', code \'' + code +
                                             '\' , can\'t be enabled ' +
                                             'individually as it\'s in ' +
                                             group + '\'s group.')
        # Look for groups activated when ms kind is a calibration
        else:
            for idx, code in enumerate(self.inChSel):
                if code in self.MS.inChannels.groups:
                    raise ValueError('Groups can\'t be calibrated. Channels ' +
                                     'must be calibrated individually.')
        # Constructing the inChannels list for the current take
        self.inChannels = MeasurementChList(kind='in')
        for idx, code in enumerate(self.inChSel):
            if code in self.MS.inChannels.groups:
                for chNum in self.MS.inChannels.groups[code]:
                    self.inChannels.append(self.MS.inChannels[chNum])
            else:
                self.inChannels.append(self.MS.inChannels[code])
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

    def run(self):
        self.measuredTake = []
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
            if self.MS.pause4Avg is True and self.MS.averages-i > 1:
                input('Paused before next average. {} left. '.format(
                      self.MS.averages - i - 1) + ' Press any key to ' +
                      'continue...')
        self.__dismember_take()
        self.runCheck = True

    def __dismember_take(self):
        # Dismember the measured SignalObjs into MeasuredThings for each
        # channel/group in inChSel
        chIndexCount = 0
        self.measuredThings = {}
        # Constructing a MeasuredThing for each element in self.inChSel
        for idx, code in enumerate(self.inChSel):
            # Empty list for the timeSignal arrays from each avarage
            SigObjs = []
            # Loop over the averages
            for avg in range(self.MS.averages):
                # Unpack timeSignal of a group or individual channel
                if code in self.MS.inChannels.groups:
                    membCount = len(self.MS.inChannels.groups[code])
                else:
                    membCount = 1
                timeSignal = \
                    self.measuredTake[avg].timeSignal[:, chIndexCount:
                                                      chIndexCount +
                                                      membCount]
                SigObj = SignalObj(signalArray=timeSignal,
                                   domain='time',
                                   samplingRate=self.MS.samplingRate,
                                   freqMin=self.MS.freqMin,
                                   freqMax=self.MS.freqMax,
                                   comment=self.MS.name + '\'s measured ' +
                                   self.kind)
                # Copying channels information from the measured SignalObj
                mapping = self.MS.inChannels.groups[code] if code \
                    in self.MS.inChannels.groups else \
                    [self.MS.inChannels[code].num]
                inChannels = ChannelsList()
                for chNum in mapping:
                    inChannels.append(self.inChannels[chNum])
                SigObj.channels = inChannels
                # Copying other properties from the measured SignalObj
                SigObj.timeStamp = self.measuredTake[avg].timeStamp
                SigObj.temp = self.measuredTake[avg].temp
                SigObj.RH = self.measuredTake[avg].RH
                SigObjs.append(SigObj)
            # Getting the inChannels for the current channel/group
            inChannels = MeasurementChList(kind='in',
                                           chList=SigObjs[0].channels)
            inChannels.copy_groups(self.MS.inChannels)
            # Constructing the MeasuredThing
            msdThng = MeasuredThing(kind=self.kind,
                                    arrayName=code,
                                    measuredSignals=SigObjs,
                                    inChannels=inChannels,
                                    outChannel=self.outChannel,
                                    position=(self.sourcePos,
                                              self.receiversPos[idx]),
                                    excitation=self.excitation)
            self.measuredThings[code] = msdThng  # Saving to the dict
            chIndexCount += membCount  # Counter for next channel/group

    # Properties

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
            raise TypeError('inChSel must be a list with codes of ' +
                            'individual channels and/or groups.')
        # if len(newChSelection) < len(self._MS.inChannels):
        #     raise ValueError('inChSel\' number of itens must be the ' +
        #                      'same as ' + self.MS.name + '\'s inChannels.')
        for item in newChSelection:
            if not isinstance(item, str):
                raise TypeError('inChSel must be a list with codes of ' +
                                'individual channels and/or groups.')
            elif item not in self.MS.inChannels.groups \
                    and item not in self.MS.inChannels:
                raise ValueError('\'{}\' isn\'t a valid channel or group.'
                                 .format(item))
        self._inChSel = newChSelection

    @property
    def outChSel(self):
        return self._outChSel

    @outChSel.setter
    def outChSel(self, newChSelection):
        if not isinstance(newChSelection, str):
            if newChSelection is None and self.kind in ['calibration',
                                                        'sourcerecalibration',
                                                        'noisefloor']:
                pass
            else:
                raise TypeError('outChSel must be a string with a valid ' +
                                'output channel code listed in '+self.MS.name +
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
                pass
            else:
                raise TypeError('Source must be a string.')
        # if newSource not in self.MS.outChannels:
        #     raise ValueError(newSource + ' doesn\'t exist in ' +
        #                      self.MS.name + '\'s outChannels.')
        self._sourcePos = newSource

    @property
    def receiversPos(self):
        return self._receiversPos

    @receiversPos.setter
    def receiversPos(self, newReceivers):
        if not isinstance(newReceivers, list):
            if newReceivers is None and self.kind in ['noisefloor',
                                                      'sourcerecalibration',
                                                      'calibration']:
                pass
            else:
                raise TypeError('Receivers must be a list of strings ' +
                                'with same itens number as inChSel.')
        if len(newReceivers) < len(self.inChSel):
            raise ValueError('Receivers\' number of itens must be the ' +
                             'same as inChSel.')
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
                pass
            else:
                raise TypeError('Excitation signal\'s name must be a string.')
        if newExcitation not in self.MS.excitationSignals:
            raise ValueError('Excitation signal doesn\'t exist in ' +
                             self.MS.name + '\'s excitationSignals')
        self._excitation = newExcitation
        return


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

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'kind={self.kind!r}, '
                f'arrayName={self.arrayName!r}, '
                f'measuredSignals={self.measuredSignals!r}, '
                f'inChannels={self.inChannels!r}, '
                f'position={self.position!r}, '
                f'excitation={self.excitation!r}, '
                f'outChannel={self.outChannel!r})')

    def __str__(self):
        str = self.kind + '_'  # Kind info
        if self.kind in ['roomir', 'sourcerecalibration']:
            str += self.position[0] + '-'  # Source position info
        if self.kind in ['roomir', 'noisefloor']:
            str += self.position[1] + '_'  # Receiver position info
        if self.kind in ['roomir', 'sourcerecalibration']:
            # outputChannel name info
            str += self.outChannel._channels[0].code + '-'
        str += self.arrayName + '_'  # input Channel/group name info
        if self.kind in ['roomir']:
            str += self.excitation  # Excitation signal info
        return str

    # Methods

    def h5save(self, h5group):
        # TO DO
        pass


class Transducer(object):

    # Magic methods

    def __init__(self, brand, model, serial, IR):
        self.brand = brand
        self.model = model
        self.serial = serial
        self.IR = IR
