#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:35:05 2019

@author: mtslazarin
"""

from pytta.classes._base import ChannelObj, ChannelsList
from pytta import generate, SignalObj
from pytta.functions import __h5_unpack as pyttah5unpck
import pytta.h5utilities as _h5
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
        for groupName, group in newComb.items():
            if not isinstance(group, tuple):
                raise TypeError('Groups of channels inside the ' +
                                'groups dict must be contained by' +
                                ' a tuple.')
            else:
                for chNum in group:
                    if chNum not in self.mapping:
                        raise ValueError('In group \''+ groupName + 
                                         '\', InChannel number ' + str(chNum) +
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
                if self[chNum2] == mChList[chNum2]:
                    groups[mChList.get_group_name(chNum2)] =\
                        mChList.get_group_membs(chNum2)
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
        self.creation_name = 'MeasurementSetup'
        self.measurementKinds = measurementKinds
        self._name = name
        self._samplingRate = samplingRate
        self._device = device
        self._noiseFloorTp = noiseFloorTp
        self._calibrationTp = calibrationTp
        self._excitationSignals = excitationSignals
        self._averages = averages
        self._pause4Avg = pause4Avg
        self._freqMin = freqMin
        self._freqMax = freqMax
        self.inChannels = inChannels
        self.outChannels = outChannels
        self._path = getcwd()+'/'+self.name+'/'
        self.modified = False

    def __repr__(self):
        return (f'{self.__class__.__name__}('
            f'name={self.name!r}, '
            f'samplingRate={self.samplingRate!r}, '
            f'device={self.device!r}, '
            f'noiseFloorTp={self.noiseFloorTp!r}, '
            f'calibrationTp={self.calibrationTp!r}, '
            f'excitationSignals={self.excitationSignals!r}, '
            f'averages={self.averages!r}, '
            f'pause4Avg={self.pause4Avg!r}, '
            f'freqMin={self.freqMin!r}, '
            f'freqMax={self.freqMax!r}, '
            f'inChannels={self.inChannels!r}, '
            f'outChannels={self.outChannels!r}, '
            f'path={self.path!r})')

    # Methods

    def h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already openned file via
        pytta.save(...).
        """
        h5group.attrs['class'] = 'MeasurementSetup'
        h5group.attrs['name'] = self.name
        h5group.attrs['samplingRate'] = self.samplingRate
        h5group.attrs['device'] = _h5.list_w_int_parser(self.device)
        h5group.attrs['noiseFloorTp'] = self.noiseFloorTp
        h5group.attrs['calibrationTp'] = self.calibrationTp
        h5group.attrs['averages'] = self.averages
        h5group.attrs['pause4Avg'] = self.pause4Avg
        h5group.attrs['freqMin'] = self.freqMin
        h5group.attrs['freqMax'] = self.freqMax
        h5group.attrs['inChannels'] = repr(self.inChannels)
        h5group.attrs['outChannels'] = repr(self.outChannels)
        # h5group.attrs['path'] = self.path
        h5group.create_group('excitationSignals')
        for name, excitationSignal in self.excitationSignals.items():
            excitationSignal.h5_save(h5group.create_group('excitationSignals' +
                                                          '/' + name))
        pass

    # Properties

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, newName):
        raise PermissionError('After a measurement initialization its name' +
                              'can\'t be changed.')

    @property
    def samplingRate(self):
        return self._samplingRate

    @samplingRate.setter
    def samplingRate(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'samplingRate can\'t be changed.')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'device can\'t be changed.')

    @property
    def noiseFloorTp(self):
        return self._noiseFloorTp

    @noiseFloorTp.setter
    def noiseFloorTp(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'noiseFloorTp can\'t be changed.')

    @property
    def calibrationTp(self):
        return self._calibrationTp

    @calibrationTp.setter
    def calibrationTp(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'calibrationTp can\'t be changed.')

    @property
    def excitationSignals(self):
        return self._excitationSignals

    @excitationSignals.setter
    def excitationSignals(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'excitationSignals can\'t be changed.')

    @property
    def averages(self):
        return self._averages

    @averages.setter
    def averages(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'averages can\'t be changed.')

    @property
    def pause4Avg(self):
        return self._pause4Avg

    @pause4Avg.setter
    def pause4Avg(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'pause4Avg can\'t be changed.')

    @property
    def freqMin(self):
        return self._freqMin

    @freqMin.setter
    def freqMin(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'freqMin can\'t be changed.')

    @property
    def freqMax(self):
        return self._freqMax

    @freqMax.setter
    def freqMax(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'freqMax can\'t be changed.')

    @property
    def inChannels(self):
        return self._inChannels

    @inChannels.setter
    def inChannels(self, newInput):
        if isinstance(newInput, MeasurementChList):
            self._inChannels = newInput
        elif isinstance(newInput, dict):
            self.inChannels = MeasurementChList(kind='in')
            for chCode, chContents in newInput.items():
                if chCode == 'groups':
                    self._inChannels.groups = chContents
                else:
                    self._inChannels.append(ChannelObj(num=chContents[0],
                                                       name=chContents[1],
                                                       code=chCode))

    @property
    def outChannels(self):
        return self._outChannels

    @outChannels.setter
    def outChannels(self, newInput):
        if isinstance(newInput, MeasurementChList):
            self._outChannels = newInput
        elif isinstance(newInput, dict):
            self._outChannels = MeasurementChList(kind='out')
            for chCode, chContents in newInput.items():
                self._outChannels.append(ChannelObj(num=chContents[0],
                                                    name=chContents[1],
                                                    code=chCode))

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, newValue):
        raise PermissionError('After a measurement initialization its ' +
                              'path can\'t be changed.')

class MeasurementData(object):
    """
    Class dedicated to manage in the hard drive the acquired data stored as
    MeasuredThing objects.

    This class don't need a h5_save method, as it saves itself into disc by
    its nature.
    """

    # Magic methods

    def __init__(self, MS, skipFileInit=False):
        # MeasurementSetup
        self.MS = MS
        self.path = self.MS.path
        # Workaround when roomir.h5_load instantiate a
        # new MeasurementData and it's already in disc.
        if skipFileInit:
            return
        # MeasurementData.hdf5 initialization
        if not exists(self.path):
            mkdir(self.path)
        if exists(self.path + 'MeasurementData.hdf5'):
            raise FileExistsError('ATTENTION!  MeasurementData for the ' +
                                  ' current measurement, ' + self.MS.name +
                                  ', already exists. Load it instead of '
                                  'overwriting.')
            # # Workaround for debugging
            # print('Deleting the existant measurement: ' + self.MS.name)
            # rmtree(self.path)
            # mkdir(self.path)
            # self.__h5_init()
        else:
            self.__h5_init()

    # Methods

    def __h5_init(self):
        """
        Method for initializating a brand new MeasurementData.hdf5 file
        """
        # Creating the MeasurementData file
        with h5py.File(self.path + 'MeasurementData.hdf5', 'w-') as f:
            f.create_group('MeasurementSetup')
            self.MS.h5_save(f['MeasurementSetup'])
        return
    
    def __h5_update(self):
        """
        Method for updating the MeasurementData.hdf5 file if MeasurementSetup
        was modified.
        """
        if self.MS.modified:
            with h5py.File(self.path + 'MeasurementData.hdf5', 'r+') as f:
                # Updating the MeasurementSetup
                del f['MeasurementSetup']
                f.create_group('MeasurementSetup')
                self.MS.h5_save(f['MeasurementSetup'])
                # Updating the MeasuredThings links
                # P.S.: Is that needed? Maybe not. Needs review
                myfiles = [f for f in listdir(self.path) if
                   isfile(join(self.path, f))]
                for fileName in myfiles:
                    if fileName.split('_')[0] in self.MS.measurementKinds:
                        if fileName.split('.')[0] not in f:
                            f[fileName] = h5py.ExternalLink(fileName + '.hdf5',
                                                            '/' + fileName)
            self.MS.modified = False
        return

    def save_take(self, MeasureTakeObj):
        if not MeasureTakeObj.runCheck:
            raise ValueError('Can\'t save an unacquired MeasuredThing. First' +
                             'you need to run the measurement through ' +
                             'TakeMeasure.run().')
        if MeasureTakeObj.saveCheck:
            raise ValueError('Can\'t save the this measurement take because ' +
                             'It has already been saved.')
        # Iterate over measuredThings
        for measuredThing in MeasureTakeObj.measuredThings.values():
            fileName = str(measuredThing)
            # Checking if any measurement with the same configs was take
            fileName = self.__number_the_file(fileName)
            # Saving the MeasuredThing to the disc
            measuredThing.creation_name = fileName
            h5_save(self.path + fileName + '.hdf5', measuredThing)
            # Update the MeasurementData.hdf5 file with the MeasuredThing link
            with h5py.File(self.path + 'MeasurementData.hdf5', 'r+') as f:
                f[fileName] = h5py.ExternalLink(fileName + '.hdf5',
                                                '/' + fileName)
        MeasureTakeObj.saveCheck = True
        # Update the measurementData.hdf5
        self.__h5_update()
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

    def __update_data(self):
        self._data = {}
        # ...
        return
    # Properties

    @property
    def get(self):
        self.__update_data()
        # TO DO
        return self._data

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
        if self.kind in ['roomir', 'noisefloor']:
            # Look for grouped channels through the individual channels
            for code in self.inChSel:
                if code not in self.MS.inChannels.groups:
                    chNum = self.MS.inChannels[code].num
                    if self.MS.inChannels.is_grouped(code):
                            group = self.MS.inChannels.get_group_name(chNum)
                            raise ValueError('Input channel number' +
                                             str(chNum) + ', code \'' + code +
                                             '\' , can\'t be enabled ' +
                                             'individually as it\'s in ' +
                                             group + '\'s group.')
        # Look for groups activated when ms kind is a calibration
        elif self.kind in ['sourcerecalibration', 'miccalibration']:
            for code in self.inChSel:
                if code in self.MS.inChannels.groups:
                    raise ValueError('Groups can\'t be calibrated. Channels ' +
                                     'must be calibrated individually.')
        # Constructing the inChannels list for the current take
        self.inChannels = MeasurementChList(kind='in')
        for code in self.inChSel:
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
                                     inChannels=self.inChannels.mapping,
                                     outChannels=self.outChannel.mapping,
                                     comment='roomir')
        # For miccalibration measurement kind
        if self.kind == 'miccalibration':
            self.measurementObject = \
                generate.measurement('rec',
                                     lengthDomain='time',
                                     timeLength=self.MS.calibrationTp,
                                     samplingRate=self.MS.samplingRate,
                                     freqMin=self.MS.freqMin,
                                     freqMax=self.MS.freqMax,
                                     device=self.MS.device,
                                     inChannels=self.inChannels.mapping,
                                     comment='miccalibration')
        # For noisefloor measurement kind
        if self.kind == 'noisefloor':
            self.measurementObject = \
                generate.measurement('rec',
                                     lengthDomain='time',
                                     timeLength=self.MS.noiseFloorTp,
                                     samplingRate=self.MS.samplingRate,
                                     freqMin=self.MS.freqMin,
                                     freqMax=self.MS.freqMax,
                                     device=self.MS.device,
                                     inChannels=self.inChannels.mapping,
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
                                     inChannels=self.inChannels.mapping,
                                     outChannels=self.outChannel.mapping,
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
            # Getting the receiver position
            if self.receiversPos is None:  # It happens for calibrations
                receiverPos = None
            else:
                receiverPos = self.receiversPos[idx]
            # Constructing the MeasuredThing
            msdThng = MeasuredThing(kind=self.kind,
                                    arrayName=code,
                                    measuredSignals=SigObjs,
                                    inChannels=inChannels,
                                    outChannel=self.outChannel,
                                    sourcePos=self.sourcePos,
                                    receiverPos=receiverPos,
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
        if newChSelection is None and self.kind in ['miccalibration',
                                                    'sourcerecalibration',
                                                    'noisefloor']:
            pass
        elif not isinstance(newChSelection, str):
            raise TypeError('outChSel must be a string with a valid ' +
                            'output channel code listed in '+self.MS.name +
                            '\'s outChannels.')
        elif newChSelection not in self.MS.outChannels:
            raise TypeError('Invalid outChSel code or name. It must be a ' +
                            'valid ' + self.MS.name + '\'s output channel.')
        self._outChSel = newChSelection

    @property
    def sourcePos(self):
        return self._sourcePos

    @sourcePos.setter
    def sourcePos(self, newSource):
        if newSource is None and self.kind in ['noisefloor',
                                               'miccalibration',
                                               'sourcerecalibration']:
            pass
        elif not isinstance(newSource, str):
            raise TypeError('Source must be a string.')
        self._sourcePos = newSource

    @property
    def receiversPos(self):
        return self._receiversPos

    @receiversPos.setter
    def receiversPos(self, newReceivers):
        if newReceivers is None and self.kind in ['noisefloor',
                                                  'sourcerecalibration',
                                                  'miccalibration']:
            pass
        elif not isinstance(newReceivers, list):
            raise TypeError('Receivers must be a list of strings ' +
                            'with same itens number as inChSel.')
        elif len(newReceivers) < len(self.inChSel):
            raise ValueError('Receivers\' number of itens must be the ' +
                             'same as inChSel.')
        else:
            for item in newReceivers:
                if item.split('R')[0] != '':
                    raise ValueError(item + 'isn\'t a receiver position. It ' +
                                     'must start with \'R\' succeeded by its' +
                                     ' number (e.g. R1).')
                else:
                    try:
                        int(item.split('R')[1])
                    except ValueError:
                        raise ValueError(item + 'isn\'t a receiver position ' +
                                         'code. It must start with \'R\' ' +
                                         'succeeded by its number (e.g. R1).')
        self._receiversPos = newReceivers
        return

    @property
    def excitation(self):
        return self._excitation

    @excitation.setter
    def excitation(self, newExcitation):
        if not isinstance(newExcitation, str):
            if newExcitation is None and self.kind in ['noisefloor',
                                                       'miccalibration']:
                pass
            elif newExcitation not in self.MS.excitationSignals:
                raise ValueError('Excitation signal doesn\'t exist in ' +
                                 self.MS.name + '\'s excitationSignals')
            else:
                raise TypeError('Excitation signal\'s name must be a string.')
        self._excitation = newExcitation
        return


class MeasuredThing(object):

    # Magic methods

    def __init__(self,
                 kind,
                 arrayName,
                 measuredSignals,
                 inChannels,
                 sourcePos=None,
                 receiverPos=None,
                 excitation=None,
                 outChannel=None):
        self.kind = kind
        self.arrayName = arrayName
        self.sourcePos = sourcePos
        self.receiverPos = receiverPos
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
                f'sourcePos={self.sourcePos!r}, '
                f'receiverPos={self.receiverPos!r}, '
                f'excitation={self.excitation!r}, '
                f'outChannel={self.outChannel!r})')

    def __str__(self):
        str = self.kind + '_'  # Kind info
        if self.kind in ['roomir']:
            str += self.sourcePos + '-'  # Source position info
        if self.kind in ['roomir', 'noisefloor']:
            str += self.receiverPos + '_'  # Receiver position info
        if self.kind in ['roomir', 'sourcerecalibration']:
            str += self.outChannel._channels[0].code + '-'  # outCh code info
        str += self.arrayName  # input Channel/group code info
        if self.kind in ['roomir']:
            str += '_' + self.excitation  # Excitation signal code info
        return str

    # Methods

    def h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already openned file via
        roomir.save(...).
        """
        h5group.attrs['class'] = 'MeasuredThing'
        h5group.attrs['kind'] = self.kind
        h5group.attrs['arrayName'] = self.arrayName
        h5group.attrs['inChannels'] = repr(self.inChannels)
        h5group.attrs['sourcePos'] = _h5.none_parser(self.sourcePos)
        h5group.attrs['receiverPos'] = _h5.none_parser(self.receiverPos)
        h5group.attrs['excitation'] = _h5.none_parser(self.excitation)
        h5group.attrs['outChannel'] = repr(self.outChannel)
        h5group.create_group('measuredSignals')
        for idx, msdSignal in enumerate(self.measuredSignals):
            msdSignal.h5_save(h5group.create_group('measuredSignals/' +
                                                   str(idx)))
        pass


def med_load(medname):
    """
    Load a measurement in progress
    """
    if not exists(medname + '/MeasurementData.hdf5'):
        raise NameError('{} measurement doens\'t exist.'.format(medname))
    load = h5_load(medname + '/MeasurementData.hdf5', skip=['MeasuredThing'])
    MS = load['MeasurementSetup']
    Data = MeasurementData(MS, skipFileInit=True)
    return MS, Data


def h5_save(fileName: str, *PyTTaObjs):
    """
    Open an hdf5 file, create groups for each PyTTa object, pass it to
    the own object that it saves itself inside the group.

    >>> roomir.h5_save(fileName, PyTTaObj_1, PyTTaObj_2, ..., PyTTaObj_n)

    """
    # Checking if filename has .hdf5 extension
    if fileName.split('.')[-1] != 'hdf5':
        fileName += '.hdf5'
    with h5py.File(fileName, 'w') as f:
        # Dict for counting equal names for correctly renaming
        objsNameCount = {}
        for idx, pobj in enumerate(PyTTaObjs):
            if isinstance(pobj, (MeasuredThing,
                                 MeasurementSetup)):
                # Check if creation_name was already used
                creationName = pobj.creation_name
                if creationName in objsNameCount:
                    objsNameCount[creationName] += 1
                    creationName += '_' + str(objsNameCount[creationName])
                else:
                    objsNameCount[creationName] = 1
                # create obj's group
                ObjGroup = f.create_group(creationName)
                # save the obj inside its group
                pobj.h5_save(ObjGroup)
            else:
                print("Only roomir objects can be saved through this" +
                      "function. Skipping object number " + str(idx) + ".")


def h5_load(fileName: str, skip: list = []):
    """
    Load a roomir hdf5 file and recreate it's objects
    """
    # Checking if the file is an hdf5 file
    if fileName.split('.')[-1] != 'hdf5':
        raise ValueError("roomir.h5_load only works with *.hdf5 files")
    f = h5py.File(fileName, 'r')
    loadedObjects = {}
    objCount = 0  # Counter for loaded objects
    totCount = 0  # Counter for total groups
    for PyTTaObjName, PyTTaObjGroup in f.items():
        totCount += 1
        if PyTTaObjGroup.attrs['class'] in skip:
            pass
        else:
            try:
                loadedObjects[PyTTaObjName] = __h5_unpack(PyTTaObjGroup)
                objCount += 1
            except TypeError:
                print('Skipping hdf5 group named {} as '.format(PyTTaObjName) +
                      'it isnt a PyTTa object group.')
    f.close()
    # Final message
    plural1 = 's' if objCount > 1 else ''
    plural2 = 's' if totCount > 1 else ''
    print('Imported {} PyTTa object-like group'.format(objCount) + plural1 +
          ' of {} group'.format(totCount) + plural2 +
          ' inside the hdf5 file.')
    return loadedObjects


def __h5_unpack(ObjGroup):
    if ObjGroup.attrs['class'] == 'MeasurementSetup':
        name = ObjGroup.attrs['name']
        samplingRate = int(ObjGroup.attrs['samplingRate'])
        device = int(_h5.list_w_int_parser(ObjGroup.attrs['device']))
        noiseFloorTp = float(ObjGroup.attrs['noiseFloorTp'])
        calibrationTp = float(ObjGroup.attrs['calibrationTp'])
        averages = int(ObjGroup.attrs['averages'])
        pause4Avg = ObjGroup.attrs['pause4Avg']
        freqMin = float(ObjGroup.attrs['freqMin'])
        freqMax = float(ObjGroup.attrs['freqMax'])
        inChannels = eval(ObjGroup.attrs['inChannels'])
        outChannels = eval(ObjGroup.attrs['outChannels'])
        excitationSignals = {}
        for sigName, excitationSignal in ObjGroup['excitationSignals'].items():
            excitationSignals[sigName] = __h5_unpack(excitationSignal)
        MS = MeasurementSetup(name,
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
                              calibrationTp)
        return MS
    elif ObjGroup.attrs['class'] == 'MeasuredThing':
        kind = ObjGroup.attrs['kind']
        arrayName = ObjGroup.attrs['arrayName']
        inChannels = eval(ObjGroup.attrs['inChannels'])
        sourcePos = _h5.none_parser(ObjGroup.attrs['sourcePos'])
        receiverPos = _h5.none_parser(ObjGroup.attrs['receiverPos'])
        excitation = _h5.none_parser(ObjGroup.attrs['excitation'])
        outChannel = _h5.none_parser(ObjGroup.attrs['outChannel'])
        if outChannel is not None:
            outChannel = eval(outChannel)
        measuredSignals = []
        for h5MsdSignal in ObjGroup['measuredSignals'].values():
            measuredSignals.append(__h5_unpack(h5MsdSignal))
        MsdThng = MeasuredThing(kind=kind,
                                arrayName=arrayName,
                                inChannels=inChannels,
                                sourcePos=sourcePos,
                                receiverPos=receiverPos,
                                outChannel=outChannel,
                                excitation=excitation,
                                measuredSignals=measuredSignals)
        return MsdThng
    else:
        return pyttah5unpck(ObjGroup)
