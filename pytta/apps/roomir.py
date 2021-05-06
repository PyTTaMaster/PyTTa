#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RooomIR application.

This application was developed using the toolbox's existent functionalities,
and is aimed to give support to acquire and post-process impulsive responses,
and calculate room acoustics parameters.

For syntax purposes you should start with:

    >>> from pytta import roomir as roomir

The functionalities comprise tools for data acquisition (MeasurementSetup,
MeasurementData, and TakeMeasure classes) and post-processing
(MeasurementPostProcess class), including some really basic statistical
treatment.

The workflow consists of creating a roomir.MeasurementSetup and
a roomir.MeasurementData object, then taking a measurement with a
roomir.TakeMeasure object (see its documentation for possible take kinds);
there will be as many TakeMeasure instantiations as measurement takes.
After a measurement is taken it's saved through the
roomir.MeasurementData.save_take() method.

All measured responses and the post-processed impulsive responses are stored
as MeasuredThing class objects, while the calculated results as Analysis class
objects. All data is stored in the HDF5 file scheme designed for the toolbox.
For more information about the specific file scheme for the
MeasurementData.hdf5 file check the MeasurementData class documentation.

It is also possible to use the LabJack U3 hardware with the EI 1050 probe to
acquire humidity and temperature values during the measurement. Check the
TakeMeasure class documentation for more information.

The usage of this app is shown in the files present in the examples folder.

Available classes:

    >>> roomir.MeasurementSetup
    >>> roomir.MeasurementData
    >>> roomir.TakeMeasure
    >>> roomir.MeasuredThing
    >>> roomir.MeasurementPostProcess

Available functions:

    >> MS, D = roomir.med_load('med-name')

For further information, check the docstrings for each class and function
mentioned above. This order is also recommended.

Authors:
    Matheus Lazarin, matheus.lazarin@eac.ufsm.br

"""

from pytta.classes._base import ChannelObj, ChannelsList
from pytta import generate, SignalObj, ImpulsiveResponse, Analysis
from pytta import rooms
from pytta.functions import __h5_unpack as pyttah5unpck
from pytta import _h5utils as _h5
import numpy as np
import scipy.stats
from scipy import interpolate
import h5py
from os import getcwd, listdir, mkdir
from os.path import isfile, join, exists
import copy as cp
import traceback

# Dict with the measurementKinds
# TO DO FUTURE: add 'inchcalibration', 'outchcalibration'
# Roomir measurement versus pytta basic i/o
measurementKinds = {'roomres': 'PlayRecMeasure',
                    'noisefloor': 'RecMeasure',
                    'miccalibration': 'RecMeasure',
                    'sourcerecalibration': 'PlayRecMeasure',
                    'channelcalibration': 'PlayRecMeasure'}

class _MeasurementChList(ChannelsList):

    # Magic methods

    def __init__(self, kind, groups={}, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the ChannelsList
        # Rest of initialization
        self.kind = kind
        self.groups = groups

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                # _MeasurementChList properties
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
                # Getting groups information for reconstructing
                # inChannels
                try:
                    if self[chNum2] == mChList[chNum2]:
                        groups[mChList.get_group_name(chNum2)] =\
                            mChList.get_group_membs(chNum2)
                except IndexError:
                    pass
        self.groups = groups

# Workaround for class name change. If removed old roomir files won't load.
MeasurementChList = _MeasurementChList

class MeasurementSetup(object):
    """Holds the measurement setup information. Managed in disc by the
    roomir.MeasurementData class and loaded back to memory through the
    roomir.med_load function.

    Creation parameters (default), (type):
    --------------------------------------

        * name (), (str):
            Measurement identification name. Used on roomir.med_load('name');

        * samplingRate (), (int):
            Device sampling rate;

        * device (), (list | int):
            Audio I/O device identification number from pytta.list_devices().
            Can be an integer for input and output with the same device, or a
            list for different devices. E.g.:

                >>> device = [1,2]  % [input,output]

        * excitationSignals (), (dict):
            Dictionary containing SignalObjs with excitation signals. E.g.:

                >>> excitSigs = {'sweep18': pytta.generate.sweep(fftDegree=18),
                                 'speech': pytta.read_wave('sabine.wav')}

        * freqMin (), (float):
            Analysis band's lower limit;

        * freqMax (), (float):
            Analysis band's upper limit

        * inChannels (), (dict):
            Dict containing input channel codes, hardware channel and name.
            Aditionally is possible to group channels with an extra 'groups'
            key. E.g.:

                >>> inChannels={'LE': (4, 'Left ear'),
                                'RE': (3, 'Right ear'),
                                'AR1': (5, 'Array mic 1'),
                                'AR2': (6, 'Array mic 2'),
                                'AR3': (7, 'Array mic 3'),
                                'Mic1': (1, 'Mic 1'),
                                'Mic2': (2, 'Mic 2'),
                                'groups': {'HATS': (4, 3),
                                           'Array': (5, 6, 7)} }

        * inCompensations (), (dict):
            Magnitude compensation for each input transducers, but not
            mandatory for all. E.g.:

                >>> inCompensations={'AR1': (AR1SensFreq, AR1SensdBMag),
                                     'AR2': (AR2SensFreq, AR2SensdBMag),
                                     'AR3': (AR3SensFreq, AR3SensdBMag),
                                     'Mic2': (M2SensFreq, M2SensdBMag) }

        * outChannels (default), (type):
            Dict containing output channel codes, hardware channel and name.
            E.g.:

                >>> outChannels={'O1': (1, 'Dodecahedron 1'),
                                 'O2': (2, 'Dodecahedron 2'),
                                 'O3': (4, 'Room sound system') }

        * outCompensations (default), (type):
            Magnitude compensation for each output transducers, but not
            mandatory for all. E.g.:

                >>> outCompensations={'O1': (Dodec1SensFreq, Dodec1SensdBMag),
                                      'O2': (Dodec2SensFreq, Dodec2SensdBMag) }

        * averages (), (int):
            Number of averages per take. This option is directly connected to
            the confidence interval calculated by the MeasurementPostProcess
            class methods. Important in case you need some statistical
            treatment;

        * pause4Avg (), (bool):
            Option for pause between averages;

        * noiseFloorTp (), (float):
            Recording time length in seconds for noisefloor measurement take
            type;

        * calibrationTp (default), (type):
            Recording time length in seconds for microphone indirect
            calibration take type;

    """

    # Magic methods

    def __init__(self,
                 name,
                 samplingRate,
                 device,
                 excitationSignals,
                 freqMin,
                 freqMax,
                 inChannels,
                 inCompensations,
                 outChannels,
                 outCompensations,
                 averages,
                 pause4Avg,
                 noiseFloorTp,
                 calibrationTp):
        self.initing = True
        self.creation_name = 'MeasurementSetup'
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
        self.inChannels = inChannels
        self.inCompensations = inCompensations
        self.outChannels = outChannels
        self.outCompensations = outCompensations
        self.path = getcwd()+'/'+self.name+'/'
        self.modified = False
        self.initing = False

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

    def _h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already opened file via
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

        h5group.create_group('inCompensations')
        for chCode, comp in self.inCompensations.items():
            h5group.create_group('inCompensations/' + chCode)
            h5group['inCompensations/' + chCode + '/freq'] = comp[0]
            h5group['inCompensations/' + chCode + '/dBmag'] = comp[1]

        h5group.create_group('outCompensations')
        for chCode, comp in self.outCompensations.items():
            h5group.create_group('outCompensations/' + chCode)
            h5group['outCompensations/' + chCode + '/freq'] = comp[0]
            h5group['outCompensations/' + chCode + '/dBmag'] = comp[1]

        h5group.create_group('excitationSignals')
        for name, excitationSignal in self.excitationSignals.items():
            excitationSignal._h5_save(h5group.create_group('excitationSignals' +
                                                          '/' + name))
        pass

    # Properties

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its name' +
                                  'can\'t be changed.')
        self._name = newValue

    @property
    def samplingRate(self):
        return self._samplingRate

    @samplingRate.setter
    def samplingRate(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'samplingRate can\'t be changed.')
        self._samplingRate = newValue

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'device can\'t be changed.')
        self._device = newValue

    @property
    def noiseFloorTp(self):
        return self._noiseFloorTp

    @noiseFloorTp.setter
    def noiseFloorTp(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'noiseFloorTp can\'t be changed.')
        self._noiseFloorTp = newValue

    @property
    def calibrationTp(self):
        return self._calibrationTp

    @calibrationTp.setter
    def calibrationTp(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'calibrationTp can\'t be changed.')
        self._calibrationTp = newValue

    @property
    def excitationSignals(self):
        return self._excitationSignals

    @excitationSignals.setter
    def excitationSignals(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'excitationSignals can\'t be changed.')
        self._excitationSignals = newValue
    @property
    def averages(self):
        return self._averages

    @averages.setter
    def averages(self, newValue):
        if not isinstance(newValue, int):
            raise TypeError("'averages' type must be int.")
        self._averages = newValue
        self.modified = True

    @property
    def pause4Avg(self):
        return self._pause4Avg

    @pause4Avg.setter
    def pause4Avg(self, newValue):
        self._pause4Avg = newValue
        self.modified = True

    @property
    def freqMin(self):
        return self._freqMin

    @freqMin.setter
    def freqMin(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'freqMin can\'t be changed.')
        self._freqMin = newValue

    @property
    def freqMax(self):
        return self._freqMax

    @freqMax.setter
    def freqMax(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'freqMax can\'t be changed.')
        self._freqMax = newValue

    @property
    def inChannels(self):
        return self._inChannels

    @inChannels.setter
    def inChannels(self, newInput):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'inChannels can\'t be changed.')
        if isinstance(newInput, _MeasurementChList):
            self._inChannels = newInput
        elif isinstance(newInput, dict):
            self._inChannels = _MeasurementChList(kind='in')
            for chCode, chContents in newInput.items():
                if chCode == 'groups':
                    self._inChannels.groups = chContents
                else:
                    self._inChannels.append(ChannelObj(num=chContents[0],
                                                       name=chContents[1],
                                                       code=chCode))

    @property
    def inCompensations(self):
        return self._inCompensations

    @inCompensations.setter
    def inCompensations(self, newComps):
        # if not self.initing:
        #     raise PermissionError('After a measurement initialization its ' +
        #                           'inCompensations can\'t be changed.')
        if isinstance(newComps, dict):
            for chCode, comp in newComps.items():
                if chCode not in self.inChannels.codes:
                    raise NameError("Channel code '{}' in ".format(chCode) +
                                    "inCompensations isn't a valid input " +
                                    "transducer.")
                if not isinstance(comp, tuple):
                    raise TypeError("inCompensations must be a dict with " +
                            "channel codes as keys and a tuple containing " +
                            "the compensation in dB and the frequency vector" +
                            " as values. Empty dict for no compensation.")
            self._inCompensations = newComps
            self.modified = True
        else:
            raise TypeError("inCompensations must be a dict with channel " +
                            "codes as keys and a tuple containing the " +
                            "compensation in dB and the frequency vector as " +
                            "values. Empty dict for no compensation.")

    @property
    def outChannels(self):
        return self._outChannels

    @outChannels.setter
    def outChannels(self, newInput):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'outChannels can\'t be changed.')
        if isinstance(newInput, _MeasurementChList):
            self._outChannels = newInput
        elif isinstance(newInput, dict):
            self._outChannels = _MeasurementChList(kind='out')
            for chCode, chContents in newInput.items():
                self._outChannels.append(ChannelObj(num=chContents[0],
                                                    name=chContents[1],
                                                    code=chCode))

    @property
    def outCompensations(self):
        return self._outCompensations

    @outCompensations.setter
    def outCompensations(self, newComps):
        # if not self.initing:
        #     raise PermissionError('After a measurement initialization its ' +
        #                           'inCompensations can\'t be changed.')
        if isinstance(newComps, dict):
            for chCode, comp in newComps.items():
                if chCode not in self.outChannels.codes:
                    raise NameError("Channel code '{}' in ".format(chCode) +
                                    "outCompensations isn't a valid output " +
                                    "transducer.")
                if not isinstance(comp, tuple):
                    raise TypeError("outCompensations must be a dict with " +
                            "channel codes as keys and a tuple containing " +
                            "the compensation in dB and the frequency vector" +
                            " as values. Empty dict for no compensation.")
            self._outCompensations = newComps
            self.modified = True
        else:
            raise TypeError("outCompensations must be a dict with channel " +
                            "codes as keys and a tuple containing the " +
                            "compensation in dB and the frequency vector as " +
                            "values. Empty dict for no compensation.")

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, newValue):
        if not self.initing:
            raise PermissionError('After a measurement initialization its ' +
                                  'path can\'t be changed.')
        self._path = newValue

class MeasurementData(object):
    """
    Class intended to store and retrieve from disc the acquired data as
    MeasuredThing objects plus the MeasurementSetup. Used to calculate the
    impulsive responses and calibrated responses as well.

    Instantiation:

        >>> MS = pytta.roomir.MeasurementSetup(...)
        >>> D = pytta.roomir.MeasurementData(MS)

    Creation parameters (default), (type):
    ---------------------------------------

        * MS (), (roomir.MeasurementSetup):
            MeasurementSetup object;

    Methods (input arguments):
    ---------------------------

        * save_take(...):
            Save an acquired roomir.TakeMeasure in disc;

        * get(...):
            Retrieves from disc MeasuredThings that matches the provided tags
            and returns a dict;

        * calculate_ir(...):
            Calculate the Impulsive Responses from the provided dict, which is
            obtained trough the get(...) method. Saves the result as new
            MeasuredThing objects;

        * calibrate_res(...):
            Apply indirect calibration to the measured signals from the
            provided dict, which is obtained trough the get(...) method. Saves
            the result as new MeasuredThing objects;

    For further information on methods see its specific documentation.

    """

    # Magic methods

    def __init__(self, MS, skipFileInit=False):
        # MeasurementSetup
        self.MS = MS
        self.path = self.MS.path
        # Workaround when roomir._h5_load instantiate a new MeasurementData
        # and it's already in disc. For roomir.load_med purposes.
        if skipFileInit:
            self._h5_update_links()
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
            # print('Deleting the existent measurement: ' + self.MS.name)
            # rmtree(self.path)
            # mkdir(self.path)
            # self._h5_init()
        else:
            self._h5_init()

    # Methods

    def _h5_init(self):
        """
        Method for initializing a brand new MeasurementData.hdf5 file
        """
        # Creating the MeasurementData file
        with h5py.File(self.path + 'MeasurementData.hdf5', 'w-') as f:
            f.create_group('MeasurementSetup')
            self.MS._h5_save(f['MeasurementSetup'])
        return

    def _h5_update_links(self):
        """
        Method for update MeasurementData.hdf5 with all MeasuredThings in disc.
        """
        with h5py.File(self.path + 'MeasurementData.hdf5', 'r+') as f:
            # Updating the MeasuredThings links
            myFiles = [file for file in listdir(self.path) if
                isfile(join(self.path, file))]
            # Check if all MeasuredThings files are linked
            for myFile in myFiles:
                if myFile.split('_')[0] in self.MS.measurementKinds:
                    if myFile.split('.')[0] not in f:
                        f[myFile] = h5py.ExternalLink(myFile + '.hdf5',
                                                      '/' + myFile)
            # Check if all MeasuredThings links' files exist
            for link in list(f):
                if link + '.hdf5' not in myFiles:
                    if link != 'MeasurementSetup':
                        del f[link]
        return

    def _h5_update_MS(self):
        """
        Method for update MeasurementSetup in MeasurementData.hdf5.
        """
        if self.MS.modified:
            # Updating the MeasurementSetup
            with h5py.File(self.path + 'MeasurementData.hdf5', 'r+') as f:
                del f['MeasurementSetup']
                f.create_group('MeasurementSetup')
                self.MS._h5_save(f['MeasurementSetup'])
                self.MS.modified = False
        return

    def _h5_link(self, newMeasuredThing=None):
        """
        Method for update MeasurementData.hdf5 with a new MeasuredThing hdf5
        file link.
        """
        with h5py.File(self.path + 'MeasurementData.hdf5', 'r+') as f:
            # Update the MeasurementData.hdf5 file with the MeasuredThing link
            if newMeasuredThing is not None:
                if isinstance(newMeasuredThing, MeasuredThing):
                        fileName = newMeasuredThing.creation_name
                        if fileName in f:
                            print('Link already exist. Updating.')
                            del f[fileName]
                        f[fileName] = h5py.ExternalLink(fileName + '.hdf5',
                                                        '/' + fileName)
                else:
                    raise TypeError('Only MeasuredThings can be updated to ' +
                                    'MeasurementData.hdf5')
            else:
                print('Skipping _h5_link as no MeasuredThing was provided.')
        return


    def save_take(self, TakeMeasureObj):
        """
        Saves in disc the resultant roomir.MeasuredThings from a
        roomir.TakeMeasure's run.

        Input arguments (default), (type):
        -----------------------------------

            * TakeMeasureObj (), (roomir.TakeMeasure)

        Usage:

            >>> myTake = roomir.TakeMeasure(kind="roomres", ...)
            >>> myTake.run()
            >>> D.save_take(myTake)

        """
        if not TakeMeasureObj.runCheck:
            if TakeMeasureObj.saveCheck:
                raise ValueError("Can't save an already saved " +
                                 "MeasuredThing. Run TakeMeasure again or " +
                                 "create a new one then run it.")
            else:
                raise ValueError('Can\'t save an unacquired MeasuredThing. First' +
                                'you need to run the measurement through ' +
                                'TakeMeasure.run().')
        if TakeMeasureObj.saveCheck:
            raise ValueError('Can\'t save the this measurement take because ' +
                             'it has already been saved. Run again or ' +
                             'configure other TakeMeasure.')
        self._h5_update_MS()
        self._h5_update_links()
        # Iterate over measuredThings
        for measuredThing in TakeMeasureObj.measuredThings.values():
            fileName = str(measuredThing)
            # Number the file checking if any measurement with the same configs
            # was take
            fileName = self._number_the_file(fileName)
            # Saving the MeasuredThing to the disc
            measuredThing.creation_name = fileName
            _h5_save(self.path + fileName + '.hdf5', measuredThing)
            # Save the MeasuredThing link to measurementData.hdf5
            self._h5_link(measuredThing)
        TakeMeasureObj.saveCheck = True
        TakeMeasureObj.runCheck = False
        return

    def _number_the_file(self, fileName):
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

    def get(self, *args, skipMsgs=False):
        """
        Gets from disc the MeasuredThings that matches with the provided
        tags as non-keyword arguments.


        Input arguments (default), (type):
        -----------------------------------

            * non-keyword arguments (), (string):

                Those are tags. Tags are the main information about the
                roomir.MeasuredThings found on its filename stored in disc.
                You can mix as many tags as necessary to define well your
                search. The possible tags are:

                    - kind (e.g. roomres, roomir... See other available kinds
                            in roomir.MeasuredThing's docstrings);

                        >>> D.get('roomres')  % Gets all measured roomres;


                    - source-receiver cfg. (single info for 'noisefloor'
                                            MeasuredThing kind, e.g. 'R3'; or
                                            a pair for other kinds, e.g.
                                            'S1-R1');

                        >>> D.get('S1')  % Gets everything measured in source
                                         % position 1


                    - output-input cfg. (single channel for 'miccalibration',
                                         e.g. 'Mic1'; output-input pair
                                         for other kinds, e.g. 'O1-Mic1'):

                        >>> D.get('O1')  % Gets everything measured through
                                         % ouput 1

                        >>> D.get('O1','Mic1')  % or
                        >>> D.get('O1-Mic1')


                    - excitation signal (e.g. 'SWP19'):

                        >>> D.get('SWP20')  % Gets everything measured with
                                            % excitation signal 'SWP20'


                    - take counter (e.g. '_1', '_2', ...):

                        >>> D.get('_1')  % Gets all first takes


                Only MeasuredThings that matches all provided tags will be
                returned.

            * skipMsgs (false), (bool):
                Don't show the search's resultant messages.


        Return (type):
        --------------

            * getDict (dict):
                Dict with the MeasuredThing's filenames as keys and the
                MeasuredThing itself as values. e.g.:

                    >>> getDict = {'roomres_S1-R1_O1-Mic1_SWP19_1':
                                       roomir.MeasuredThing}


        Specific usage example:

            Get the 'roomir' MeasuredThing's first take at S1-R1 with Mic1,
            Output1, and sweep:

            >>> getDict = MeasurementData.get('roomrir', 'Mic1-O1', '_1', ...
                                              'SWP19', 'S1-R1')

            As you see, the order doesn't matter because the algorithm just
            look for matching tags in the filenames.

        """
        if not skipMsgs:
            print('Finding the MeasuredThings with {} tags.'.format(args))
        # Get MeasuredThings from disc
        myFiles = [f for f in listdir(self.path) if
                   isfile(join(self.path, f))]
        myFiles.pop(myFiles.index('MeasurementData.hdf5'))

        # Filtering files with the provided tags
        filteredFiles = []
        for fileName in myFiles:
            append = True
            for arg in args:
                if arg not in fileName:
                    append = False
            if append:
                filteredFiles.append(fileName)
        if not skipMsgs:
            print('Found {} matching MeasuredThings'.format(len(filteredFiles)))
        # Retrieving the MeasuredThings
        msdThngs = {}
        for idx, fileName in enumerate(filteredFiles):
            if not skipMsgs:
                print('Loading match {}: {}'.format(idx+1, fileName))
            loadedThing = _h5_load(self.path + fileName, skipMsgs=True)
            msdThngName = fileName.split('.')[0]
            msdThngs[msdThngName] = loadedThing[msdThngName]
        if not skipMsgs:
            print('Done.')
        return msdThngs

    def calculate_ir(self, getDict,
                     calibrationTake=1,
                     skipInCompensation=False,
                     skipOutCompensation=False,
                     skipBypCalibration=False,
                     skipRegularization=False,
                     skipIndCalibration=False,
                     IREndManualCut=None,
                     IRStartManualCut=None,
                     skipSave=False,
                     whereToOutComp='excitation'):
        """
        Gets the dict returned from the roomir.MeasuremenData.get() method,
        calculate the impulsive responses, store to disc, and return the
        correspondent getDict. Check the input arguments below for options.

        This method generates new MeasuredThings with a kind derived from
        the measured kind. The possible conversions are:

            - 'roomres' MeasuredThing kind to a 'roomir' MeasuredThing kind;
            - 'channelcalibration' to 'channelcalibir';
            - 'sourcerecalibration' to 'sourcerecalibir' (for the Strengh
                                                          Factor recalibration
                                                          method. See
                                                          pytta.rooms.G for
                                                          more information);


        Input arguments (default), (type):
        ----------------------------------

            * getDict (), (dict):
                Dict from the roomir.MeasurementData.get(...) method;

            * calibrationTake (1), (int):
                Choose the take from the 'miccalibration' MeasuredThing
                for the indirect calibration of the correspondent input
                channels;

            * skipInCompensation (False), (bool):
                Option for skipping compensation on the input chain with the
                provided response to the MeasurementSetup;

            * skipOutCompensation (False), (bool):
                Option for skipping compensation on the output chain with the
                provided response to the MeasurementSetup;

            * skipBypCalibration (False), (bool):
                Option for skipping bypass calibration. Bypass calibration
                means deconvolving with the impulsive response measured between the
                output and input of the soundcard.

            * skipRegularization (False), (bool):
                Option for skipping Kirkeby's regularization. For more
                information see pytta.ImpulsiveResponse's docstrings.

            * skipIndCalibration (False), (bool):
                Option for skipping the indirect calibration;

            * IREndManualCut (None), (float):
                Manual cut of the calculated impulsive response;

            * IRStartManualCut (None), (float):
                Manual cut of the calculated impulsive response;

            * skipSave (False), (bool):
                Option to skip saving the new MeasuredThings to disc.
                Usefull when you need to calculated the same impulsive response
                with different options and don't want to override the one saved
                previously.


        Return (type):
        --------------

            * getDict (dict):
                Dict with the calculated MeasuredThings, with filenames as keys
                and the MeasuredThing itself as values. e.g.:

                    >>> getDict = {'roomir_S1-R1_O1-Mic1_SWP19_1':
                                       roomir.MeasuredThing}

        """
        self._h5_update_MS()
        self._h5_update_links()
        IRMsdThngs = {}
        for msdThngName, msdThng in getDict.items():
            print("Calculating impulsive " +
                  "response for '{}'".format(msdThngName))
            if not isinstance(msdThng, MeasuredThing):
                raise TypeError("'roomir.calculate_ir' only works with " +
                                "MeasuredThing objects.")
            elif msdThng.kind not in ['roomres',
                                      'sourcerecalibration',
                                      'channelcalibration']:
                print("-- Impulsive responses can only be calculated " +
                      "from a MeasuredThing of 'roomres', " +
                      "'sourcerecalibration' or 'channelcalibration' kind.")
                continue

            # Getting the excitation signal
            kind = msdThng.kind
            origExcitationTimeSig = \
                cp.copy(self.MS.excitationSignals[msdThng.excitation].
                    timeSignal)
            origExctSamplingRate = \
                self.MS.excitationSignals[msdThng.excitation].samplingRate
            timeSigWGain = origExcitationTimeSig*msdThng.outputLinearGain
            excitationWGain = SignalObj(signalArray=timeSigWGain,
                                        domain='time',
                                        samplingRate=origExctSamplingRate)

            # Calculate the IRs
            IRs = []

            if kind in ['channelcalibration']:
                skipIndCalibration = True
                skipBypCalibration = True
                skipInCompensation = True
                skipOutCompensation = True
                skipRegularization = True
                print("- Skipping calibrations and compensations as it's a " +
                        "channel calibration IR.")

            # Apply compensation for output transducer
            if not skipOutCompensation and whereToOutComp == 'excitation':
                outChCode = msdThng.outChannel.codes[0]
                print("-- Applying compensation to the output " +
                        "signal for output '{}'.".format(outChCode))
                if outChCode not in self.MS.outCompensations:
                    print("--- No compensation found for output " +
                            "channel " +
                            "'{}'. ".format(outChCode) +
                            "Skipping compensation on this " +
                            "channel.")
                else:
                    excitFreqVector = \
                        excitationWGain.freqVector
                    excitFreqSignal = \
                        excitationWGain.freqSignal[:,0]
                    excitdBMag = \
                        20*np.log10(np.abs(excitFreqSignal))
                    outTransSensFreq = \
                        self.MS.outCompensations[outChCode][0]
                    outTransSensdBMag = \
                        self.MS.outCompensations[outChCode][1]
                    interp_func = \
                        interpolate.interp1d(outTransSensFreq,
                                                outTransSensdBMag,
                                                fill_value= \
                                                (outTransSensdBMag[0],
                                                outTransSensdBMag[-1]),
                                                bounds_error=False)
                    interpOutTransSensdBMag = \
                        interp_func(excitFreqVector)
                    correctedExcitdBMag = \
                        excitdBMag + interpOutTransSensdBMag
                    correctedExcitFreqSignal = \
                        10**(correctedExcitdBMag/20)
                    r = correctedExcitFreqSignal
                    teta = np.angle(excitFreqSignal)
                    correctedExcitFreqSignal = \
                        r*np.cos(teta) + r*np.sin(teta)*1j
                    excitationWGain.freqSignal = \
                        correctedExcitFreqSignal
            else:
                print("-- Skipping output transducer compensation on " +
                      "excitation signal.")

            for avg in range(msdThng.averages):
                print('- Calculating average {}'.format(avg+1))

                recording = msdThng.measuredSignals[avg]

                # Apply compensation for input transducer
                if not skipInCompensation:
                    newFreqSignal = np.zeros(recording.freqSignal.shape,
                                            dtype=np.complex64)
                    for chIndex in range(msdThng.numChannels):
                        inChCode = msdThng.inChannels.codes[chIndex]
                        outChCode = msdThng.outChannel.codes[0]
                        print("-- Applying compensation for the input " +
                                "transducer '{}'.".format(inChCode))
                        if inChCode not in self.MS.inCompensations:
                            print("--- No compensation found for input " +
                                    "channel " +
                                    "'{}'. ".format(inChCode) +
                                    "Skipping compensation on this " +
                                    "channel.")
                            newFreqSignal[:, chIndex] = \
                                recording.freqSignal[:, chIndex]
                        else:
                            roomResFreqVector = recording.freqVector
                            roomResFreqSignal = recording.freqSignal[:,chIndex]
                            roomResdBMag = \
                                20*np.log10(np.abs(roomResFreqSignal))

                            inTransSensFreq = \
                                self.MS.inCompensations[inChCode][0]
                            inTransSensdBMag = \
                                self.MS.inCompensations[inChCode][1]
                            in_interp_func = \
                                interpolate.interp1d(inTransSensFreq,
                                                     inTransSensdBMag,
                                                     fill_value= \
                                                        (inTransSensdBMag[0],
                                                        inTransSensdBMag[-1]),
                                                     bounds_error=False)
                            interpInTransSensdBMag = \
                                in_interp_func(roomResFreqVector)

                            if not skipOutCompensation and \
                                 whereToOutComp == 'recording':
                                print("-- Applying compensation to the " +
                                       "recording signal for output " +
                                       "'{}'.".format(outChCode))
                                outTransSensFreq = \
                                    self.MS.outCompensations[outChCode][0]
                                outTransSensdBMag = \
                                    self.MS.outCompensations[outChCode][1]
                                out_interp_func = \
                                    interpolate.interp1d(outTransSensFreq,
                                                            outTransSensdBMag,
                                                            fill_value= \
                                                            (outTransSensdBMag[0],
                                                            outTransSensdBMag[-1]),
                                                            bounds_error=False)
                                interpOutTransSensdBMag = \
                                    out_interp_func(roomResFreqVector)

                                correctedRoomResdBMag = \
                                    roomResdBMag - interpInTransSensdBMag - \
                                        interpOutTransSensdBMag
                            else:
                                correctedRoomResdBMag = \
                                    roomResdBMag - interpInTransSensdBMag

                            correctedRoomResFreqSignal = \
                                10**(correctedRoomResdBMag/20)
                            r = correctedRoomResFreqSignal
                            teta = np.angle(roomResFreqSignal)
                            correctedRoomResFreqSignal = \
                                r*np.cos(teta) + r*np.sin(teta)*1j
                            newFreqSignal[:,chIndex] = \
                                correctedRoomResFreqSignal

                    recording.freqSignal = newFreqSignal
                else:
                    print("-- Skipping input transducer compensation.")

                if skipRegularization:
                    regularization = False
                    print("-- Skipping Kirkeby IR regularization.")
                else:
                    regularization = True

                IR = ImpulsiveResponse(excitation=excitationWGain,
                                       recording=recording,
                                       regularization=regularization)

                # Applying bypass calibration to in/out channel
                if not skipBypCalibration:
                    newFreqSignal = np.zeros(IR.systemSignal.freqSignal.shape,
                                            dtype=np.complex64)
                    # bypFreqSignal = np.ones(IR.systemSignal.freqSignal.shape,
                    #                         dtype=np.complex64)
                    for chIndex in range(msdThng.numChannels):
                        inChCode = msdThng.inChannels.codes[chIndex]
                        outChCode = msdThng.outChannel.codes[0]
                        chFreqSignal = IR.systemSignal.freqSignal[:, chIndex]
                        chSignal = SignalObj(chFreqSignal, 'freq',
                                             self.MS.samplingRate)
                        # chSignal = IR.systemSignal
                        print("-- Applying the bypass calibration on" +
                                " '{}' channel.".format(inChCode))
                        # Get the channelcalibir signal
                        chCalibThngs = [calib for calib in
                                        self.get('channelcalibir',
                                                    inChCode,
                                                    outChCode,
                                                    skipMsgs=True).values()]
                        if len(chCalibThngs) == 0:
                            print("--- No channelcalibir found for input/" +
                                    "output channels " +
                                    "'{}/{}'. ".format(inChCode,outChCode) +
                                    "Skipping channel calibration on this " +
                                    "channels.")
                            newFreqSignal[:, chIndex] = \
                                IR.systemSignal.freqSignal[:, chIndex]
                        else:
                            # Getting the bypass IR
                            chCalibThng = chCalibThngs[calibrationTake-1]
                            chCalibIR = chCalibThng.measuredSignals[
                                            chCalibThng.averages//2]. \
                                                systemSignal

                            # Normalize with 1000.00 [Hz] spectrum magnitude
                            idx1k = \
                                np.where(chCalibIR.freqVector>=1000)[0][0]
                            chCalibIR.freqSignal = chCalibIR._freqSignal / \
                                float(np.abs(chCalibIR.freqSignal[idx1k]))

                            # Deconvolution
                            newIR = \
                                ImpulsiveResponse(recording=chSignal,
                                                excitation=chCalibIR,
                                                regularization=False)
                            newFreqSignal[:, chIndex] = \
                                newIR.systemSignal.freqSignal[:, 0]
                    IR.systemSignal.freqSignal = newFreqSignal
                else:
                    print("-- Skipping the bypass calibration.")

                # Applying input indirect calibration
                for chIndex in range(msdThng.numChannels):
                    inChCode = msdThng.inChannels.codes[chIndex]
                    outChCode = msdThng.outChannel.codes[0]

                    if not skipIndCalibration:
                        print("-- Applying the input indirect calibration on" +
                                " '{}' channel.".format(inChCode))
                        # Get the miccalibration signal
                        calibThngs = [calib for calib in
                                        self.get('miccalibration', inChCode,
                                                skipMsgs=True).values()]
                        if len(calibThngs) == 0:
                            print("--- No miccalibration found for channel " +
                                    "'{}'. Skipping ".format(inChCode) +
                                    "calibration in this channel.")
                        else:
                            calib = calibThngs[calibrationTake-1]. \
                                measuredSignals[calibThngs[calibrationTake-1].
                                    averages//2]
                            IR.systemSignal.calib_pressure(chIndex, calib, 1, 1000)
                    else:
                        print("-- Skipping the input indirect calibration on" +
                                " '{}' channel.".format(inChCode))

                # Copying channel names and codes
                for idx, chNum in \
                    enumerate(IR.systemSignal.channels.mapping):
                    IR.systemSignal.channels[chNum].name = \
                        msdThng.inChannels.names[idx]
                    IR.systemSignal.channels[chNum].code = \
                        msdThng.inChannels.codes[idx]

                # Cutting the IR
                if IRStartManualCut is not None or IREndManualCut is not None:
                    IREndManualCut = \
                        'end' if IREndManualCut is None else IREndManualCut
                    IRStartManualCut = \
                        0 if IRStartManualCut is None else IRStartManualCut

                    IR.systemSignal.crop(IRStartManualCut, IREndManualCut)

                IRs.append(IR)

            # Construct the MeasuredThing
            print('- Constructing the new MeasuredThing.')
            if kind == 'roomres':
                newKind = 'roomir'
            elif kind == 'sourcerecalibration':
                newKind = 'recalibir'
            elif kind == 'channelcalibration':
                newKind = 'channelcalibir'
            IRMsdThng = MeasuredThing(kind=newKind,
                                      arrayName=msdThng.arrayName,
                                      sourcePos=msdThng.sourcePos,
                                      receiverPos=msdThng.receiverPos,
                                      excitation=msdThng.excitation,
                                      measuredSignals=IRs,
                                      tempHumids=msdThng.tempHumids,
                                      timeStamps=msdThng.timeStamps,
                                      inChannels=msdThng.inChannels,
                                      outChannel=msdThng.outChannel,
                                      outputAmplification=msdThng.
                                        outputAmplification)
            # Saving
            fileName = msdThngName
            # Number the file checking if any measurement with the same configs
            # was take
            fileName = fileName.replace(kind, newKind)
            if not skipSave:
                print("-- Saving '{}'".format(fileName))
                # Saving the MeasuredThing to the disc
                IRMsdThng.creation_name = fileName
                _h5_save(self.path + fileName + '.hdf5', IRMsdThng)
                # Save the MeasuredThing link to measurementData.hdf5
                self._h5_link(IRMsdThng)
            IRMsdThngs[fileName] = IRMsdThng
        print('Done.')

        return IRMsdThngs

    def calibrate_res(self, getDict, calibrationTake=1,
                      skipInCompensation=False, skipSave=False):
        """
        Gets the dict returned from the roomir.MeasuremenData.get() method,
        apply the indirect calibration, store to disc, and return the
        correspondent getDict. Check the input arguments below for options.

        This method generates new MeasuredThings with a kind derived from
        the measured kind. The possible conversions are:

            - 'roomres' MeasuredThing kind to a 'calibrated-roomres';
            - 'noisefloor' to 'calibrated-noisefloor';
            - 'sourcerecalibration' to 'calibrated-sourcerecalibration';


        Input arguments (default), (type):
        ----------------------------------

            * getDict (), (dict):
                Dict from the roomir.MeasurementData.get(...) method;

            * calibrationTake (1), (int):
                Choose the take from the 'miccalibration' MeasuredThing
                for the indirect calibration of the correspondent input
                channels;

            * skipInCompensation (False), (bool):
                Option for skipping compensation on the input chain with the
                provided response to the MeasurementSetup;

            * skipSave (False), (bool):
                Option to skip saving the new MeasuredThings to disc.


        Return (type):
        --------------

            * getDict (dict):
                Dict with the calculated MeasuredThings, with filenames as keys
                and the MeasuredThing itself as values. e.g.:

                    >>> getDict = {'calibrated-roomres_S1-R1_O1-Mic1_SWP19_1':
                                       roomir.MeasuredThing}

        """

        CalibMsdThngs = {}
        for msdThngName, msdThng in getDict.items():
            print("Calibrating room " +
                  "response for '{}'".format(msdThngName))
            if not isinstance(msdThng, MeasuredThing):
                raise TypeError("'roomir.calibrate_res' only works with " +
                                "MeasuredThing objects.")
            elif msdThng.kind not in ['roomres', 'noisefloor',
                                      'sourcerecalibration']:
                print("-- Calibrated room response can only be calculated " +
                      "from a MeasuredThing of 'roomres', 'noisefloor', or " +
                      "'sourcerecalibration' kind")
                continue
            kind = msdThng.kind

            # Calibrate the SignalObjs
            SigObjs = []
            for avg in range(msdThng.averages):
                print('- Calculating average {}'.format(avg+1))
                # IR = ImpulsiveResponse(excitation=excitationWGain,
                #                        recording=msdThng.measuredSignals[avg])

                SigObj = cp.copy(msdThng.measuredSignals[avg])

                # Copying channel names and codes
                for idx, chNum in \
                    enumerate(SigObj.channels.mapping):
                    SigObj.channels[chNum].name = \
                        msdThng.inChannels.names[idx]
                    SigObj.channels[chNum].code = \
                        msdThng.inChannels.codes[idx]

                # Apply compensation for input transducer
                if not skipInCompensation:
                    newFreqSignal = np.zeros(SigObj.freqSignal.shape,
                                            dtype=np.complex64)
                    for chIndex in range(msdThng.numChannels):
                        inChCode = msdThng.inChannels.codes[chIndex]
                        print("-- Applying compensation for the input " +
                                "transducer '{}'.".format(inChCode))
                        if inChCode not in self.MS.inCompensations:
                            print("--- No compensation found for input " +
                                    "channel " +
                                    "'{}'. ".format(inChCode) +
                                    "Skipping compensation on this " +
                                    "channel.")
                        else:
                            roomResFreqVector = SigObj.freqVector
                            roomResFreqSignal = SigObj.freqSignal[:,chIndex]
                            roomResdBMag = \
                                20*np.log10(np.abs(roomResFreqSignal))
                            inTransSensFreq = \
                                self.MS.inCompensations[inChCode][0]
                            inTransSensdBMag = \
                                self.MS.inCompensations[inChCode][1]
                            interp_func = \
                                interpolate.interp1d(inTransSensFreq,
                                                     inTransSensdBMag,
                                                     fill_value= \
                                                        (inTransSensdBMag[0],
                                                        inTransSensdBMag[-1]),
                                                     bounds_error=False)
                            interpInTransSensdBMag = \
                                interp_func(roomResFreqVector)
                            correctedRoomResdBMag = \
                                roomResdBMag - interpInTransSensdBMag
                            correctedRoomResFreqSignal = \
                                10**(correctedRoomResdBMag/20)
                            r = correctedRoomResFreqSignal
                            teta = np.angle(roomResFreqSignal)
                            correctedRoomResFreqSignal = \
                                r*np.cos(teta) + r*np.sin(teta)*1j
                            newFreqSignal[:,chIndex] = \
                                correctedRoomResFreqSignal
                    SigObj.freqSignal = newFreqSignal
                else:
                    print("-- Skipping input transducer compensation.")

                # Apply calibration for each channel

                for chIndex in range(msdThng.numChannels):
                    # Get the miccalibration signal
                    inChCode = msdThng.inChannels.codes[chIndex]
                    print("-- Applying the input calibration on " +
                            "'{}' channel.".format(inChCode))
                    calibThngs = [calib for calib in
                                    self.get('miccalibration', inChCode,
                                            skipMsgs=True).values()]
                    if len(calibThngs) == 0:
                        print("--- No miccalibration found for channel " +
                                "'{}'. Skipping ".format(inChCode) +
                                "calibration in this channel.")
                        continue
                    calib = calibThngs[calibrationTake-1]. \
                        measuredSignals[calibThngs[calibrationTake-1].
                            averages//2]
                    SigObj.calib_pressure(chIndex, calib, 1, 1000)

                SigObjs.append(SigObj)

            # Construct the MeasuredThing
            print('- Constructing the new MeasuredThing.')
            newKind = 'calibrated-' + kind
            CalibMsdThng = MeasuredThing(kind=newKind,
                                      arrayName=msdThng.arrayName,
                                      sourcePos=msdThng.sourcePos,
                                      receiverPos=msdThng.receiverPos,
                                      excitation=msdThng.excitation,
                                      measuredSignals=SigObjs,
                                      tempHumids=msdThng.tempHumids,
                                      timeStamps=msdThng.timeStamps,
                                      inChannels=msdThng.inChannels,
                                      outChannel=msdThng.outChannel,
                                      outputAmplification=msdThng.
                                        outputAmplification)
            # Saving
            fileName = msdThngName
            # Number the file checking if any measurement with the same configs
            # was take
            fileName = fileName.replace(kind, newKind)
            if not skipSave:
                print("-- Saving '{}'".format(fileName))
                # Saving the MeasuredThing to the disc
                CalibMsdThng.creation_name = fileName
                _h5_save(self.path + fileName + '.hdf5', CalibMsdThng)
                # Save the MeasuredThing link to measurementData.hdf5
                self._h5_link(CalibMsdThng)
            CalibMsdThngs[fileName] = CalibMsdThng
        print('Done.')

        return CalibMsdThngs


class TakeMeasure(object):
    """
    Class intended to hold a measurement take configuration, take the
    measurement, and obtain the MeasuredThing for each channel/group in inChSel
    property.


    Creation parameters (default), (type):
    --------------------------------------

        * MS (), (roomir.MeasurementSetup):
            The setup of the current measurement;

        * tempHumid (), (pytta.classes.lju3ei1050):
            Object for communication with the LabJack U3 with the probe EI1050
            for acquiring temperature and humidity. For more information check
            pytta.classes.lju31050 docstrings;

        * kind (), (string):
            Possible measurement kinds:

                - 'roomres': room response to the excitation signal;
                - 'noisefloor': acquire noise floor for measurement quality
                                analysis;
                - 'miccalibration': acquire calibrator signal (94dB SPL @ 1kHz)
                                    for indirect calibration;
                - 'channelcalibration': acquire response of the ouput connected
                                        to the input channel;
                - 'sourcerecalibration': acquire recalibration response for
                                         Strength Factor measurement. For more
                                         information check
                                         pytta.rooms.strength_factor doctrings.

        * inChSel (), (list):
            Active input channels (or groups) for the current take.
            E.g.:

                >>> inChSel = ['Mic2', 'AR2']

        * receiversPos (), (list):
            List with the positions of each input channel/group.
            E.g.:

                >>> receiversPos = ['R2', 'R4']

        * excitation (), (string):
            Code of the excitation signal provided to the MeasurementSetup.
            E.g.:

                >>> excitation = 'SWP19'

        * outChSel (), (string):
            Code of the output channel provided to the MeasurementSetup. E.g.:

                >>> outChSel = 'O1'

        * outputAmplification (0) (float):
            Output amplification in dB;


        * sourcePos (), (string):
            Source's position. E.g.:

                >>> sourcePos = 'R1'


    Methods (input arguments):
    --------------------------

        * run():
            Acquire data;


    Properties (type):
    -----------

        * measuredThings (list):
            Contains the MeasuredThing objects resultant from the
            measurement take.

    """
    # TO DO: add @property .measuredThings

    # Magic methods

    def __init__(self,
                 MS,
                 kind,
                 inChSel,
                 receiversPos=None,
                 excitation=None,
                 outChSel=None,
                 outputAmplification=0,
                 sourcePos=None,
                 tempHumid=None):
        self.MS = MS
        self.tempHumid = tempHumid
        self.kind = kind
        self.inChSel = inChSel
        self.receiversPos = receiversPos
        self.excitation = excitation
        self.outChSel = outChSel
        self.outputAmplification = outputAmplification
        self.sourcePos = sourcePos
        self.__cfg_channels()
        self.__cfg_measurement_object()
        self.runCheck = False
        self.saveCheck = False

    # Methods

    def __cfg_channels(self):
        # Check for disabled combined channels
        if self.kind in ['roomres', 'noisefloor']:
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
        elif self.kind in ['sourcerecalibration',
                           'miccalibration',
                           'channelcalibration']:
            for code in self.inChSel:
                if code in self.MS.inChannels.groups:
                    raise ValueError('Groups can\'t be calibrated. Channels ' +
                                     'must be calibrated individually.')
        # Constructing the inChannels list for the current take
        self.inChannels = _MeasurementChList(kind='in')
        for code in self.inChSel:
            if code in self.MS.inChannels.groups:
                for chNum in self.MS.inChannels.groups[code]:
                    self.inChannels.append(self.MS.inChannels[chNum])
            else:
                self.inChannels.append(self.MS.inChannels[code])
        # Getting groups information for reconstructing
        # inChannels _MeasurementChList
        self.inChannels.copy_groups(self.MS.inChannels)
        # Setting the outChannel for the current take
        self.outChannel = _MeasurementChList(kind='out')
        if self.kind in ['roomres',
                         'sourcerecalibration',
                         'channelcalibration']:
            self.outChannel.append(self.MS.outChannels[self.outChSel])

    def __cfg_measurement_object(self):
        # For roomres measurement kind
        if self.kind == 'roomres':
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
                                     outputAmplification=
                                        self.outputAmplification,
                                     comment='roomres')
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
                                     outputAmplification=
                                        self.outputAmplification,
                                     comment='sourcerecalibration')
        # For sourcerecalibration measurement kind
        if self.kind == 'channelcalibration':
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
                                     outputAmplification=
                                        self.outputAmplification,
                                     comment='channelcalibration')

    def run(self):
        """
        Take measurement (initializate acquisition).

        Has no input arguments.

        Usage:

            >>> myTake.run()
            >>> D.save_take(myTake)

        """
        if self.runCheck:
            print('Overwriting previous unsaved take!')
        self.measuredTake = []
        if self.tempHumid is not None:
            self.tempHumid.start()
        for i in range(0, self.MS.averages):
            self.measuredTake.append(self.measurementObject.run())
            # Adquire do LabJack U3 + EI1050 a temperatura e
            # umidade relativa instantânea
            if self.tempHumid is not None:
                self.measuredTake[i].temp, self.measuredTake[i].RH = \
                    self.tempHumid.read()
            else:
                self.measuredTake[i].temp, self.measuredTake[i].RH = \
                    (0, 0)
            if self.MS.pause4Avg is True and self.MS.averages-i > 1:
                input('Paused before next average. {} left. '.format(
                      self.MS.averages - i - 1) + ' Press any key to ' +
                      'continue...')
        if self.tempHumid is not None:
            self.tempHumid.stop()
        self.__dismember_take()
        self.runCheck = True
        self.saveCheck = False

    def __dismember_take(self):
        # Dismember the measured SignalObjs into MeasuredThings for each
        # channel/group in inChSel
        chIndexCount = 0
        self.measuredThings = {}
        # Constructing a MeasuredThing for each element in self.inChSel
        for idx, code in enumerate(self.inChSel):
            # Empty list for the timeSignal arrays from each average
            SigObjs = []
            # Empty list for the temperature and rel. humidity from each avg
            tempHumids = []
            # Empty list for the timeStamps
            timeStamps = []
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
                timeStamps.append(self.measuredTake[avg].timeStamp)
                tempHumids.append((self.measuredTake[avg].temp,
                                   self.measuredTake[avg].RH))
                SigObjs.append(SigObj)
            # Getting the inChannels for the current channel/group
            inChannels = _MeasurementChList(kind='in',
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
                                    timeStamps=timeStamps,
                                    tempHumids=tempHumids,
                                    inChannels=inChannels,
                                    outChannel=self.outChannel,
                                    outputAmplification=
                                        self.outputAmplification,
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
    def outputAmplification(self):
        return self._outputAmplification

    @outputAmplification.setter
    def outputAmplification(self, newOutputGain):
        if not isinstance(newOutputGain, (float, int)):
            raise TypeError("'outputAmplification must be float or int.")
        self._outputAmplification = newOutputGain
        return

    @property
    def sourcePos(self):
        return self._sourcePos

    @sourcePos.setter
    def sourcePos(self, newSource):
        if newSource is None and self.kind in ['noisefloor',
                                               'miccalibration',
                                               'sourcerecalibration',
                                               'channelcalibration']:
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
                                                  'miccalibration',
                                                  'channelcalibration']:
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
    """
    Obtained through a roomir.TakeMeasure object. Contains information of a
    measurement take for one source-receiver configuration. Shouldn't be
    instantiated by user.

    Properties (type):
    ------------------

        * kind (str):
            Possible kinds for a MeasuredThing:

                - 'roomres';
                - 'roomir' ('roomres' after IR calculation);
                - 'noisefloor';
                - 'miccalibration'
                - 'sourcerecalibration';
                - 'recalibir' ('sourcerecalibration' after IR calculation);
                - 'channelcalibration';
                - 'channelcalibir' ('channelcalibration' after IR calculation);

        * arrayName (str):
            Code of the input channel or group (array);

        * measuredSignals (list):
            Contains the resultant SignalObjs or ImpulsiveResponses;

        * timeStamps (list):
            Contains the timestamps for each measurement take;

        * tempHumids (list):
            Contains the temperature and humidity readings for each measurement
            take;

        * inChannels (roomir._MeasurementChList):
            Measurement channel list object. Identifies the used soundcard's
            input channels;

        * sourcePos (str):
            Source position;

        * receiverPos (str):
            Receiver's (microphone or array) position;

        * excitation (str):
            Excitation signal code in MeasurementSetup;

        * outChannel (roomir._MeasurementChList):
            Measurement channel list object. Identifies the used soundcard's
            output channel;

        * outputAmplification (float):
            Output amplification in dB set for the take;

        * outputLinearGain (float):
            Output amplification in linear scale;

        * numChannels (float):
            The number of channels;

        * averages (int):
            The number of averages;


    """

    # Magic methods

    def __init__(self,
                 kind='',
                 arrayName='',
                 measuredSignals=[],
                 timeStamps=[], # with default because compatibility issues
                 tempHumids=[],  # with default because compatibility issues
                 inChannels=None,
                 sourcePos=None,
                 receiverPos=None,
                 excitation=None,
                 outChannel=None,
                 outputAmplification=0):
        self.kind = kind
        self.arrayName = arrayName
        self.sourcePos = sourcePos
        self.receiverPos = receiverPos
        self.excitation = excitation
        self.measuredSignals = measuredSignals
        self.timeStamps = timeStamps
        self.tempHumids = tempHumids
        self.inChannels = inChannels
        self.outChannel = outChannel
        self.outputAmplification = outputAmplification

    # Magic methods

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'kind={self.kind!r}, '
                f'arrayName={self.arrayName!r}, '
                f'measuredSignals={self.measuredSignals!r}, '
                f'timeStamps={self.timeStamps!r}, '
                f'tempHumids={self.tempHumids!r}, '
                f'inChannels={self.inChannels!r}, '
                f'sourcePos={self.sourcePos!r}, '
                f'receiverPos={self.receiverPos!r}, '
                f'excitation={self.excitation!r}, '
                f'outChannel={self.outChannel!r}, '
                f'outputAmplification={self.outputAmplification!r})')

    def __str__(self):
        str = self.kind + '_'  # Kind info
        if self.kind in ['roomres', 'roomir']:
            str += self.sourcePos + '-'  # Source position info
        if self.kind in ['roomres', 'roomir', 'noisefloor']:
            str += self.receiverPos + '_'  # Receiver position info
        if self.kind in ['roomres', 'roomir',
                         'sourcerecalibration', 'recalibir',
                         'channelcalibration', 'channelcalibir']:
            str += self.outChannel._channels[0].code + '-'  # outCh code info
        str += self.arrayName  # input Channel/group code info
        if self.kind in ['roomres', 'roomir',
                         'sourcerecalibration', 'recalibir',
                         'channelcalibration', 'channelcalibir']:
            str += '_' + self.excitation  # Excitation signal code info
        return str

    # Methods

    def _h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already opened file via
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
        h5group.attrs['outputAmplification'] = self.outputAmplification
        h5group.attrs['timeStamps'] = self.timeStamps
        h5group['tempHumids'] = self.tempHumids
        h5group.create_group('measuredSignals')
        for idx, msdSignal in enumerate(self.measuredSignals):
            msdSignal._h5_save(h5group.create_group('measuredSignals/' +
                                                   str(idx)))
        pass

    # Properties

    @property
    def numChannels(self):
        try:
            numChannels = self.measuredSignals[0].timeSignal.shape[1]
        except IndexError:
            numChannels = 1
        return numChannels


    @property
    def averages(self):
        return len(self.measuredSignals)

    @property
    def outputLinearGain(self):
        return 10**(self.outputAmplification/20)


def _mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


class MeasurementPostProcess(object):
    """
    Holds a measurement post processing session.

    Creation parameters (default), (type):
    --------------------------------------

        * nthOct (3), (int):
            Number of bands per octave;

        * minFreq (100), (float):
            Minimum analysis' frequency;

        * maxFreq (1000), (float):
            Minimum analysis' frequency;


    Methods:
    --------

        * RT (getDict, decay, IREndManualCut)
            Calculates the average reverberation time for each source-receiver
            pair from the provided getDict. Also calculates a 95% confidence
            interval from a T-Student distribution, which is dependent on the
            number of averages. For more information on reverberation time
            calculation go to pytta.rooms.reverberation_time

        * G (Lpe_avgs, Lpe_revCh, V_revCh, T_revCh, Lps_revCh, Lps_inSitu):
            Calculates the strength factor from the G terms provided by several
            getDict-like dicts from other G methods of this class.

        * G_Lpe_inSitu (roomirsGetDict, IREndManualCut):
            Calculates the sound exposure level of the room impulsive
            response. For more information about this G term go to
            pytta.rooms.G_Lpe;


        * G_Lpe_revCh(roomirsGetDict, IREndManualCut):
            Calculates the sound exposure level of the reverberation chamber's
            impulsive response. For more information about this G term go to
            pytta.rooms.G_Lpe;

        * G_Lps (sourcerecalibirGetDict):
            Calculates the sound exposure level of the recalibration impulsive
            response. For more information about this G term go to
            pytta.rooms.G_Lps;


        * G_T_revCh (roomirsGetDict, IREndManualCut, T):
            Calculates the mean reverberation time of the reverberation chamber;


    For further information check the specific method's docstrings.

    """

    def __init__(self, nthOct=3, minFreq=100, maxFreq=10000):
        self.nthOct = nthOct
        self.minFreq = minFreq
        self.maxFreq = maxFreq

    def RT(self, roomirsGetDict, decay=20, IREndManualCut=None):
        """
        Calculates the average reverberation time for each source-receiver
        pair from the dict provided by the roomir.MeasurementData.get method.

        Also calculates a 95% confidence interval from a T-Student
        distribution, which is dependent on the  number of averages. For more
        information on reverberation time calculation go to
        pytta.rooms.reverberation_time.

        Parameters (default), (type):
        ------------------------------

            * getDict ('Time [s]'), (str):
                a dict from the roomir.MeasurementData.get method containing
                MeasuredThings of the type 'roomir' (room impulsive response);

            * decay (20), (int):
                dynamic range of the line fit;

            * IREndManualCut (None), (float):
                remove the end of the impulsive response from IREndManualCut,
                given in seconds;


        Return (type):
        --------------

            * TR (dict):
                a dict containing the mean reverberation time (Analysis) for
                each source-receiver configuration, which is a key;

        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        # TR in all positions and averagßes
        # TR_avgs = {'S1R1': [Analysis_avg1, An_avg2, ..., An_avgn]}
        TR_avgs = {}
        roomirs = roomirsGetDict
        for roomir in roomirs.values():
            roomir.sourcePos = \
                'Sx' if roomir.sourcePos is None else roomir.sourcePos
            roomir.receiverPos = \
                'Rx' if roomir.receiverPos is None else roomir.receiverPos
            SR = roomir.sourcePos + roomir.receiverPos
            if SR not in TR_avgs:
                TR_avgs[SR] = []
            for IR in roomir.measuredSignals:
                TR_avgs[SR].append(rooms.analyse(IR, 'RT', decay,
                                                 nthOct=self.nthOct,
                                                 minFreq=self.minFreq,
                                                 maxFreq=self.maxFreq,
                                                 IREndManualCut=IREndManualCut))

        # Statistics for TR
        TR_CI = {}
        for SR, TRs in TR_avgs.items():
            TR_CI[SR] = []
            if len(TRs) < 2:
                TR_CI[SR] = None
                continue
            data = np.vstack([an.data for an in TRs])
            for bandIdx in range(data.shape[1]):
                TR_CI[SR].append(_mean_confidence_interval(data[:,bandIdx])[1])

        # Calculate mean TR value
        TR = {}
        for SR, TRs in TR_avgs.items():
            TRs[0].anType = 'L'
            sum4avg = TRs[0]
            for TRan in TRs[1:]:
                TRan.anType = 'L'
                sum4avg += TRan
            meanTR = sum4avg/len(TRs)
            meanTR.errorLabel = 'Confiança 95% dist. T-Student'
            meanTR.error = TR_CI[SR]
            meanTR.dataLabel = SR
            TR[SR] = meanTR

        TR['dictName'] = creation_name
        return TR

    def G_Lps(self, recalibirsGetDict):
        """
        Calculates the mean sound exposure level from the recalibration
        impulsive responses. For more information about this G term go to
        pytta.rooms.G_Lps;

        Parameters (default), (type):
        -----------------------------

            * G_Lps (), (dict from rmr.D.get(...)):
                a dict from the roomir.MeasurementData.get method containing
                MeasuredThings of the type 'recalibir';

        Return (type):
        --------------

            * finalLps (Analysis):
                an Analysis object with the averaged recalibration exposure
                level;

        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        recalibirs = recalibirsGetDict
        # Calculating Lps for each recalibir MeasuredThing
        Lps_avgs = {}
        for name, recalibir in recalibirs.items():
            # Calculating average Lps from MeasuredThing averages
            avgs = []
            for IR in recalibir.measuredSignals:
                avg = rooms.G_Lps(IR, self.nthOct, self.minFreq, self.maxFreq)
                avg.anType = 'L'
                avgs.append(avg)
            Lps_avgs[name] = avgs
        # Statistics for Lps
        Lps_CI = {}
        for name, Lpss in Lps_avgs.items():
            data = np.vstack([an.data for an in Lpss])
            Lps_CI[name] = []
            for bandIdx in range(data.shape[1]):
                Lps_CI[name].append(
                    _mean_confidence_interval(data[:,bandIdx])[1])
        # Calculate average Lps
        finalLpss = {}
        for name, Lpss in Lps_avgs.items():
            sum4avg = Lpss[0]
            LpsWindowLimits = [Lpss[0].windowLimits, Lpss[1].windowLimits]
            for Lps in Lpss[1:]:
                sum4avg += Lps
                LpsWindowLimits.append(Lps.windowLimits)
            Lps = sum4avg/len(Lpss)
            Lps.errorLabel = 'Confiança 95% dist. T-Student'
            Lps.error = Lps_CI[name]
            Lps.dataLabel = name
            Lps.creation_name = 'Lps_' + name
            Lps.windowLimits = LpsWindowLimits
            finalLpss[name] = Lps
        finalLpss['dictName'] = creation_name
        return finalLpss

    def G_Lpe_inSitu(self, roomirsGetDict, IREndManualCut=None):
        """
        Calculates the room impulsive response' sound exposure level for each
        source-receiver cfg. For more information about this G term go to
        pytta.rooms.G_Lpe;

        Receives

        Parameters (default), (type):
        -----------------------------

            * roomirsGetDict (), ():
                a dict from the roomir.MeasurementData.get method containing
                MeasuredThings of the type 'roomir' (room impulsive response);

            * IREndManualCut (None), (float):
                remove the end of the impulsive response from IREndManualCut,
                given in seconds;

        Return (type):
        --------------

            * Lpe_avgs (dict):
                a dict containing a list with the sound exposure level averages
                (Analyses) for each source-receiver configuration, which is a
                key;

        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        #  Lpe in all positions and averagßes
        # Lpe_avgs = {'S1R1': [Analysis_avg1, An_avg2, ..., An_avgn]}
        roomirs = roomirsGetDict
        Lpe_avgs = {}
        for roomir in roomirs.values():
            roomir.sourcePos = \
                'Sx' if roomir.sourcePos is None else roomir.sourcePos
            roomir.receiverPos = \
                'Rx' if roomir.receiverPos is None else roomir.receiverPos
            SR = roomir.sourcePos + roomir.receiverPos
            if SR not in Lpe_avgs:
                Lpe_avgs[SR] = []
            for IR in roomir.measuredSignals:
                Lpe_avgs[SR].append(rooms.G_Lpe(IR, self.nthOct, self.minFreq,
                                                self.maxFreq, IREndManualCut))
        Lpe_avgs['dictName'] = creation_name
        return Lpe_avgs

    def G_Lpe_revCh(self, roomirsGetDict, IREndManualCut=None):
        """
        Calculates the mean sound exposure level of the reverberation chamber's
        impulsive response. For more information about this G term go to
        pytta.rooms.G_Lpe;


        Parameters (default), (type):
        -----------------------------

            * roomirsGetDict (), ():
                a dict from the roomir.MeasurementData.get method containing
                MeasuredThings of the type 'roomir' (room impulsive response);

            * IREndManualCut (None), (float):
                remove the end of the impulsive response from IREndManualCut,
                given in seconds;

        Return (type):
        --------------

            * Lpe (Analysis):
                an Analysis with the mean sound exposure level calculated from
                all the reverberation chamber's impulsive responses;

        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        SigObjs = []
        if isinstance(roomirsGetDict, dict):
            for msdThng in roomirsGetDict.values():
                SigObjs.extend([IR.systemSignal for IR in msdThng.measuredSignals])
        elif isinstance(roomirsGetDict, list):
            SigObjs = roomirsGetDict

        Lpe_avgs = []
        for IR in SigObjs:
            Lpe_avgs.append(rooms.G_Lpe(IR, self.nthOct, self.minFreq,
                                        self.maxFreq, IREndManualCut))

        Leq = 0
        for L in Lpe_avgs:
            Leq =  L + Leq
        Lpe = Leq / len(Lpe_avgs)
        Lpe.anType = 'mixed'
        Lpe.unit = 'dB'
        Lpe.creation_name = creation_name

        return Lpe

    def G(self, Lpe_avgs, Lpe_revCh, V_revCh, T_revCh, Lps_revCh, Lps_inSitu):
        """
        Calculates the mean strength factor for each source-receiver
        configuration with the G terms provided by other methods of this class.
        Also provides some basic statistical treatment.

        For further information on the recalibration method (to correct
        changes on the source's sound power) check:

            Christensen, C. L.; Rindel, J. H. APPLYING IN-SITU RECALIBRATION
            FOR SOUND STRENGTH MEASUREMENTS IN AUDITORIA.

        Parameters (default), (type):
        ------------------------------

            * Lpe_avgs (), (dict from rmr.get(...)):
                a dict provided by the rmr.get(...) method. Calculates a mean
                G for all source-receiver configurations provided with the
                dict. Also calculates the 95% confidence interval for a
                T-Student distribution;


            * Lpe_revCh (), (Analysis):
                a pytta.Analysis object with the mean exposure level inside
                the reverberation chamber during the source calibration (sound
                power measurement);

            * V_revCh (), (float):
                the volume of the reverberation chamber;

            * T_revCh (), (Analysis):
                a pytta.Analysis object for the reverberation chamber's
                reverberation time;

            * Lps_revCh (), (Analysis)
                the exposure level of the recalibration procedure in the
                reverberation chamber;

            * Lps_inSitu (), (Analysis):
                the exposure level of the recalibration procedure in situ;

         Return (type):
        --------------

            * G (dict):
                a dict containing the mean G (Analysis) for each
                source-receiver configuration, which is a key;

        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        # G (omfg) calculation
        # G averages in all positions
        # G_avgs = {'S1R1': [Analysis_avg1, Analysis_avg2, ..., Analysis_avgn]}
        V_revCh = 207
        G_avgs = {}
        for SR, Lpes in Lpe_avgs.items():
            if 'S' in SR and 'R' in SR:
                G_avgs[SR] = []
                for Lpe in Lpes:
                    G_avgs[SR].append(rooms.strength_factor(Lpe,
                                                            Lpe_revCh,
                                                            V_revCh, T_revCh,
                                                            Lps_revCh,
                                                            Lps_inSitu))
        # Statistics for G
        G_CI = {}
        for SR, Gs in G_avgs.items():
            G_CI[SR] = []
            if len(Gs) < 2:
                G_CI[SR] = None
                continue
            data = np.vstack([an.data for an in Gs])
            for bandIdx in range(data.shape[1]):
                G_CI[SR].append(_mean_confidence_interval(data[:,bandIdx])[1])
        # Calculate mean G value
        G = {}
        for SR, Gs in G_avgs.items():
            Gs[0].anType = 'L'
            sum4avg = Gs[0]
            for Gan in Gs[1:]:
                Gan.anType = 'L'
                sum4avg += Gan
            meanG = sum4avg/len(Gs)
            meanG.errorLabel = 'Confiança 95% dist. T-Student'
            meanG.error = G_CI[SR]
            meanG.dataLabel = SR
            G[SR] = meanG
        G['dictName'] = creation_name
        return G

    def G_T_revCh(self, roomirsGetDict, IREndManualCut=None, T=20):
        """
        Calculates the mean reverberation time of the reverberation chamber;

        Parameters (default), (type):
        -----------------------------

            * roomirsGetDict (), ():
                a dict from the roomir.MeasurementData.get method containing
                MeasuredThings of the type 'roomir' (room impulsive response);

            * IREndManualCut (None), (float):
                remove the end of the impulsive response from IREndManualCut,
                given in seconds;

        Return (type):
        --------------

            * T_revCh (Analysis):
                an Analysis with the mean reverberation time calculated from
                all the reverberation chamber's impulsive responses;

        """
        # Code snippet to guarantee that generated object name is
        # the declared at global scope
        # for frame, line in traceback.walk_stack(None):
        for framenline in traceback.walk_stack(None):
            # varnames = frame.f_code.co_varnames
            varnames = framenline[0].f_code.co_varnames
            if varnames == ():
                break
        # creation_file, creation_line, creation_function, \
        #     creation_text = \
        extracted_text = \
            traceback.extract_stack(framenline[0], 1)[0]
            # traceback.extract_stack(frame, 1)[0]
        # creation_name = creation_text.split("=")[0].strip()
        creation_name = extracted_text[3].split("=")[0].strip()

        roomirs = roomirsGetDict
        RTs = []
        for msdThngName, msdThng in roomirs.items():
            print("Calculating RTs for {}".format(msdThngName))
            for idx, avg in enumerate(msdThng.measuredSignals):
                print("Calculating average {}".format(idx))
                sigObj = avg.systemSignal
                TR = rooms.analyse(sigObj, 'RT', T, nthOct=self.nthOct,
                                   minFreq=self.minFreq,
                                   maxFreq=self.maxFreq,
                                   plotLundebyResults=False,
                                   IREndManualCut=IREndManualCut)
                RTs.append(TR)
        # Averaging in space
        bands = RTs[0].bands
        spacialAvgdRT = []
        for bandIdx, _ in enumerate(bands):
            bandRTsum = 0
            bandRTcount = 0
            for RTan in RTs:
                RT = RTan.data[bandIdx]
                if np.isnan(RT):
                    continue
                else:
                    bandRTcount += 1
                    bandRTsum += RT
            bandRTmean = bandRTsum / bandRTcount
            spacialAvgdRT.append(bandRTmean)
        # Constructing the Analysis
        T_revCh = Analysis(anType='RT', nthOct=self.nthOct,
                           minBand=float(bands[0]), maxBand=float(bands[-1]),
                           data=spacialAvgdRT)
        T_revCh.creation_name = creation_name
        return T_revCh

def med_load(medname):
    """
    Loads a measurement to continue measuring or either post processing.

    Usage:

        >>> MS, D = roomir.med_load('measurement name')

    Parameters (defualt), (type):
    -----------------------------

        * medname (), (str):
            the measurement name given in the MeasurementSetup object
            instantiation;

    Return (type):
    --------------

        * (roomir.MeasurementSetup, roomir.MeasurementData) (tuple)

    """
    if not exists(medname + '/MeasurementData.hdf5'):
        raise NameError('{} measurement doesn\'t exist.'.format(medname))
    print('Loading the MeasurementSetup from MeasurementData.hdf5.')
    load = _h5_load(medname + '/MeasurementData.hdf5', skip=['MeasuredThing'])
    MS = load['MeasurementSetup']
    D = MeasurementData(MS, skipFileInit=True)
    return MS, D


def _h5_save(fileName: str, *PyTTaObjs):
    """
    Open an hdf5 file, create groups for each PyTTa object, pass it to
    the own object that it saves itself inside the group.

    >>> roomir._h5_save(fileName, PyTTaObj_1, PyTTaObj_2, ..., PyTTaObj_n)

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
                pobj._h5_save(ObjGroup)
            else:
                print("Only roomir objects can be saved through this" +
                      "function. Skipping object number " + str(idx) + ".")


def _h5_load(fileName: str, skip: list = [], skipMsgs: bool = False):
    """_h5_load

    >>> roomir._h5_load('file_1.hdf5', skip=['MeasuredThing'], skipMsgs=False)

    Load a roomir hdf5 file and recreate it's objects

    :param fileName: file's name
    :type fileName: str
    :param skip: list with object types to skip in load, defaults to []
    :type skip: list, optional
    :param skipMsgs: don't show the load messages, defaults to False
    :type skipMsgs: bool, optional

    :return: dictionary with the loaded objects
    :rtype: dict
    """

    # "a""
    #
    # ""a"
    # Checking if the file is an hdf5 file
    if fileName.split('.')[-1] != 'hdf5':
        raise ValueError("roomir._h5_load only works with *.hdf5 files")
    f = h5py.File(fileName, 'r')
    loadedObjects = {}
    objCount = 0  # Counter for loaded objects
    totCount = 0  # Counter for total groups
    for PyTTaObjName, PyTTaObjGroup in f.items():
        totCount += 1
        try:
            if PyTTaObjGroup.attrs['class'] in skip:
                pass
            else:
                try:
                    loadedObjects[PyTTaObjName] = _h5_unpack(PyTTaObjGroup)
                    objCount += 1
                except TypeError:
                    if not skipMsgs:
                        print("Skipping hdf5 group named {} as "
                            .format(PyTTaObjName) +
                            "it isn't a PyTTa object group.")
        except AttributeError:
            if not skipMsgs:
                print('Skipping {} as its link is broken.'.format(PyTTaObjName))
    f.close()
    # Final message
    if not skipMsgs:
        plural1 = 's' if objCount > 1 else ''
        plural2 = 's' if totCount > 1 else ''
        print('Imported {} PyTTa object-like group'.format(objCount) +
             plural1 + ' of {} group'.format(totCount) + plural2 +
            ' inside the hdf5 file.')
    return loadedObjects


def _h5_unpack(ObjGroup):
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
            excitationSignals[sigName] = _h5_unpack(excitationSignal)
        inCompensations = {}
        if 'inCompensations' in ObjGroup:
            for chCode, group in ObjGroup['inCompensations'].items():
                inCompensations[chCode] = (np.array(group['freq']),
                                           np.array(group['dBmag']))
        outCompensations = {}
        if 'outCompensations' in ObjGroup:
            for chCode, group in ObjGroup['outCompensations'].items():
                outCompensations[chCode] = (np.array(group['freq']),
                                            np.array(group['dBmag']))
        MS = MeasurementSetup(name,
                              samplingRate,
                              device,
                              excitationSignals,
                              freqMin,
                              freqMax,
                              inChannels,
                              inCompensations,
                              outChannels,
                              outCompensations,
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
        # Added with an if for compatibility issues
        if 'outputAmplification' in ObjGroup.attrs:
            outputAmplification = ObjGroup.attrs['outputAmplification']
        else:
            outputAmplification = 0
        if 'tempHumids' in ObjGroup:
            tempHumids = [tuple(arr) for arr in list(ObjGroup['tempHumids'])]
        else:
            tempHumids = []
        if 'timeStamps' in ObjGroup.attrs:
            timeStamps = list(ObjGroup.attrs['timeStamps'])
        else:
            timeStamps = []
        if outChannel is not None:
            outChannel = eval(outChannel)
        measuredSignals = []
        for h5MsdSignal in ObjGroup['measuredSignals'].values():
            measuredSignals.append(_h5_unpack(h5MsdSignal))
        MsdThng = MeasuredThing(kind=kind,
                                arrayName=arrayName,
                                inChannels=inChannels,
                                sourcePos=sourcePos,
                                receiverPos=receiverPos,
                                outChannel=outChannel,
                                excitation=excitation,
                                measuredSignals=measuredSignals,
                                tempHumids=tempHumids,
                                timeStamps=timeStamps,
                                outputAmplification=outputAmplification)
        return MsdThng
    else:
        return pyttah5unpck(ObjGroup)
