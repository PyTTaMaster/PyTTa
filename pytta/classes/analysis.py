# -*- coding: utf-8 -*-

from pytta.classes._instanceinfo import RememberInstanceCreationInfo as RICI
from pytta.classes.filter import fractional_octave_frequencies as FOF
from pytta.classes import SignalObj, OctFilter, ImpulsiveResponse
from pytta.utils import fractional_octave_frequencies as FOF, freq_to_band
from math import isnan
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import locale
from pytta import _h5utils as _h5
from pytta import _plot as plot
import copy as cp


# Analysis types and its units
anTypes = {'RT': ('s', 'Reverberation time'),
           'C': ('dB', 'Clarity'),
           'D': ('%', 'Definition'),
           'G': ('dB', 'Strength factor'),
           'L': ('dB', 'Level'),
           'mixed': ('-', 'Mixed')}

class Analysis(RICI):
    """
    Objects belonging to the Analysis class holds fractional octave band data.

    It does conveniently the operations linearly between Analyses of the type
    'Level'. Therefore those operations do not occur with values in dB scale.

    Available Analysis' types below.

    For more information see each parameter/attribute/method specific
    documentation.

    Creation parameters (default), (type):
    --------------------------------------

        * anType (), (string):
            Type of the Analysis. May be:
                - 'RT' for 'Reverberation time' Analysis in [s];
                - 'C' for 'Clarity' in dB;
                - 'D' for 'Definition' in %;
                - 'G' for 'Strength factor' in dB;
                - 'L' for any 'Level' Analysis in dB (e.g: SPL);
                - 'mixed' for any combination between the types above.

        * nthOct, (int):
            The number of fractions per octave;

        * minBand, (int | float):
            The exact or approximated start frequency;

        * maxBand, (int | float):
            The exact or approximated stop frequency;

        * data, (list | numpy array):
            The data with the exact number of bands between the specified minimum
            (minBand) and maximum band (maxBand);

        * dataLabel (''), (string):
            Label for plots;

        * error, (list | numpy array):
            The error with the exact number of bands between the specified
            minimum (minBand) and maximum band (maxBand);

        * errorLabel (''), (string):
            Label for plots;

        * comment ('No comments.'), (string):
            Some comment about the object.

        * xLabel (None), (string):
            x axis plot label;

        * yLabel (None), (string):
            y axis plot label;

        * title (None), (string):
            plot title.


    Attributes:
    -----------

        * bands (NumPy array):
            The bands central frequencies.


    Properties:
    -----------

        * minBand, (int | float):
            When a new limit is set data is automatic adjusted.

        * maxBand, (int | float):
            When a new limit is set data is automatic adjusted.


    Methods:
    --------

        * plot_bars():
            Generates a bar plot.

    """

    # Magic methods

    def __init__(self,
                anType,
                nthOct,
                minBand,
                maxBand,
                data,
                dataLabel=None,
                error=None,
                errorLabel='Error',
                comment='No comments.',
                xLabel=None,
                yLabel=None,
                title=None):
        super().__init__()
        self.anType = anType
        self.nthOct = nthOct
        self._minBand = minBand
        self._maxBand = maxBand
        self.data = data
        self.dataLabel = dataLabel
        self.error = error
        self.errorLabel = errorLabel
        self.comment = comment
        # Plot infos memory
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.title = title
        return

    def __str__(self):
        return ('1/{} octave band {} '.format(self.nthOct, self.anType) +
            'analysis from the {} [Hz] to the '.format(self.minBand) +
            '{} [Hz] band.'.format(self.maxBand))

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'anType={self.anType!r}, '
                f'nthOct={self.nthOct!r}, '
                f'minBand={self.minBand!r}, '
                f'maxBand={self.maxBand!r}, '
                f'data={self.data!r}, '
                f'comment={self.comment!r})')

    def __add__(self, other):
        if isinstance(other, Analysis):
            if other.range != self.range:
                raise ValueError("Can't subtract! Both Analysis have" +
                                    " different band limits.")
            if self.anType == 'L':
                if other.anType == 'L':
                    data = []
                    for idx, value in enumerate(self.data):
                        d = 10*np.log10(10**(value/10) +
                                        10**(other.data[idx]/10))
                        data.append(d)
                    anType = 'L'
                elif other.anType in ['mixed', 'C', 'D', 'RT']:
                    data = self.data + other.data
                    anType = 'mixed'
                else:
                    raise NotImplementedError("Operation not implemented " +
                                              "for Analysis types " +
                                              anTypes[self.anType][1] +
                                              " and " +
                                              anTypes[other.anType][1] +
                                              ".")
            else:
                data = self.data + other.data
                anType = 'mixed'
        elif isinstance(other, (int, float)):
            if self.anType == 'L':
                data = [10*np.log10(10**(dt/10) + 10**(other/10))
                        for dt in self.data]
                anType = 'L'
            else:
                data = self.data + other
                anType = 'mixed'
        else:
            raise NotImplementedError("Operation not implemented between " +
                                      "Analysis and {}".format(type(other)) +
                                      "types.")
        selfDataLabel = self.dataLabel if self.dataLabel is not None \
            else 'Analysis 1'
        if hasattr(other,'dataLabel'):
            if other.dataLabel is not None:
                otherDataLabel = other.dataLabel
            else:
                otherDataLabel = 'Analysis 2'
        else:
            otherDataLabel = 'Analysis 2'
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data, dataLabel=selfDataLabel +
                                            ' + ' + otherDataLabel,
                        error=None, errorLabel=None,
                        comment=None,
                        xLabel=self.xLabel, yLabel=self.yLabel,
                        title=None)

        return result

    def __sub__(self, other):
        if isinstance(other, Analysis):
            if other.range != self.range:
                raise ValueError("Can't subtract! Both Analysis have" +
                                    " different band limits.")
            if self.anType == 'L':
                if other.anType == 'L':
                    data = []
                    for idx, value in enumerate(self.data):
                        d = 10*np.log10(10**(value/10) -
                                        10**(other.data[idx]/10))
                        data.append(d)
                    anType = 'L'
                elif other.anType in ['mixed', 'C', 'D', 'RT']:
                    data = self.data - other.data
                    anType = 'mixed'
                else:
                    raise NotImplementedError("Operation not implemented " +
                                              "for Analysis types " +
                                              anTypes[self.anType][1] +
                                              " and " +
                                              anTypes[other.anType][1] +
                                              ".")
            else:
                data = self.data - other.data
                anType = 'mixed'
        elif isinstance(other, (int, float)):
            if self.anType == 'L':
                data = [10*np.log10(10**(dt/10) - 10**(other/10))
                        for dt in self.data]
                anType = 'L'
            else:
                data = self.data - other
                anType = 'mixed'
        else:
            raise NotImplementedError("Operation not implemented between " +
                                      "Analysis and {}".format(type(other)) +
                                      "types.")
        selfDataLabel = self.dataLabel if self.dataLabel is not None \
            else 'Analysis 1'
        if hasattr(other,'dataLabel'):
            if other.dataLabel is not None:
                otherDataLabel = other.dataLabel
            else:
                otherDataLabel = 'Analysis 2'
        else:
            otherDataLabel = 'Analysis 2'
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data, dataLabel=selfDataLabel +
                                            ' - ' + otherDataLabel,
                        error=None, errorLabel=None,
                        comment=None,
                        xLabel=self.xLabel, yLabel=self.yLabel,
                        title=None)

        return result

    def __mul__(self, other):
        if isinstance(other, Analysis):
            if other.range != self.range:
                raise ValueError("Can't multiply! Both Analysis have " +
                                "different band limits.")
            anType='mixed'
            data=self.data*other.data
        elif isinstance(other, (int, float)):
            anType='mixed'
            data=self.data*other
        else:
            raise TypeError("Analysis can only be operated with int, float, " +
                            "or Analysis types.")
        selfDataLabel = self.dataLabel if self.dataLabel is not None \
            else 'Analysis 1'
        if hasattr(other,'dataLabel'):
            if other.dataLabel is not None:
                otherDataLabel = other.dataLabel
            else:
                otherDataLabel = 'Analysis 2'
        else:
            otherDataLabel = 'Analysis 2'
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data, dataLabel=selfDataLabel +
                                            ' * ' + otherDataLabel,
                        error=None, errorLabel=None,
                        comment=None,
                        xLabel=self.xLabel, yLabel=self.yLabel,
                        title=None)
        return result

    def __rtruediv__(self, other):
        if isinstance(other, Analysis):
            if self.anType == 'L':
                if other.range != self.range:
                    raise ValueError("Can't divide! Both Analysis have" +
                                     " different band limits.")
                elif other.anType in ['mixed', 'C', 'D', 'RT']:
                    data = other.data / self.data
                    anType = 'mixed'
                else:
                    raise NotImplementedError("Operation not implemented " +
                                              "for Analysis types " +
                                              anTypes[self.anType][1] +
                                              " and " +
                                              anTypes[other.anType][1] +
                                              ".")
            else:
                data = other.data / self.data
                anType = 'mixed'
        elif isinstance(other, (int, float)):
            if self.anType == 'L':
                data = [10*np.log10(10**(dt/10) / other)
                        for dt in self.data]
                anType = 'L'
            else:
                data = other / self.data
                anType = 'mixed'
        else:
            raise NotImplementedError("Operation not implemented between " +
                                      "Analysis and {}".format(type(other)) +
                                      "types.")
        selfDataLabel = self.dataLabel if self.dataLabel is not None \
            else 'Analysis 1'
        if hasattr(other,'dataLabel'):
            if other.dataLabel is not None:
                otherDataLabel = other.dataLabel
            else:
                otherDataLabel = 'Analysis 2'
        else:
            otherDataLabel = 'Analysis 2'
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data, dataLabel=selfDataLabel +
                                            ' / ' + otherDataLabel,
                        error=None, errorLabel=None,
                        comment=None,
                        xLabel=self.xLabel, yLabel=self.yLabel,
                        title=None)
        return result


    def __truediv__(self, other):
        if isinstance(other, Analysis):
            if self.anType == 'L':
                if other.range != self.range:
                    raise ValueError("Can't divide! Both Analysis have" +
                                     " different band limits.")
                elif other.anType in ['mixed', 'C', 'D', 'RT']:
                    data = self.data / other.data
                    anType = 'mixed'
                else:
                    raise NotImplementedError("Operation not implemented " +
                                              "for Analysis types " +
                                              anTypes[self.anType][1] +
                                              " and " +
                                              anTypes[other.anType][1] +
                                              ".")
            else:
                data = self.data / other.data
                anType = 'mixed'
        elif isinstance(other, (int, float)):
            if self.anType == 'L':
                data = [10*np.log10(10**(dt/10) / other)
                        for dt in self.data]
                anType = 'L'
            else:
                data = self.data / other
                anType = 'mixed'
        else:
            raise NotImplementedError("Operation not implemented between " +
                                      "Analysis and {}".format(type(other)) +
                                      "types.")
        selfDataLabel = self.dataLabel if self.dataLabel is not None \
            else 'Analysis 1'
        if hasattr(other,'dataLabel'):
            if other.dataLabel is not None:
                otherDataLabel = other.dataLabel
            else:
                otherDataLabel = 'Analysis 2'
        else:
            otherDataLabel = 'Analysis 2'
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data, dataLabel=selfDataLabel +
                                            ' / ' + otherDataLabel,
                        error=None, errorLabel=None,
                        comment=None,
                        xLabel=self.xLabel, yLabel=self.yLabel,
                        title=None)

        return result

    # Properties

    @property
    def anType(self):
        """Type of the Analysis.

        May be:
            - 'RT' for 'Reverberation time' Analysis in [s];
            - 'C' for 'Clarity' in dB;
            - 'D' for 'Definition' in %;
            - 'G' for 'Strength factor' in dB;
            - 'L' for any 'Level' Analysis in dB (e.g: SPL);
            - 'mixed' for any combination between the types above.

        Return:
        -------

            string.
        """
        return self._anType

    @anType.setter
    def anType(self, newType):
        if type(newType) is not str:
            raise TypeError("anType parameter makes reference to the " +
                            "calculated parameter, e.g. 'RT' for " +
                            "reverberation time, and must be a str value.")
        elif newType not in anTypes:
            raise ValueError(newType + " type not supported. May be 'RT, " +
                             "'C', 'D', 'G', 'L', or 'mixed'.")
        self.unit = anTypes[newType][0]
        self.anName = anTypes[newType][1]
        self._anType = newType
        return

    @property
    def nthOct(self):
        """octave band fraction.

        Could be 1, 3, 6...

        Return:
        -------

            int.
        """
        return self._nthOct

    @nthOct.setter
    def nthOct(self, new):
        if not isinstance(new, int):
            raise TypeError("Number of bands per octave must be int")
        if '_nthOct' in locals():
            if self.nthOct > new:
                raise TypeError("It's impossible to convert from " +
                                "{} to {} bands".format(self.nthOct, new) +
                                "per octave")
            else:
                raise NotImplementedError('Conversion between different ' +
                                          'nthOct not implemented yet.')
        else:
            self._nthOct = new
        return


    @property
    def minBand(self):
        """minimum octave fraction band.

        When a new limit is set data is automatic adjusted.

        Return:
        -------

            float.
        """
        return self._minBand

    @minBand.setter
    def minBand(self, new):
        if type(new) is not int and type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        if new in self.bands:
            print("ATTENTION! Deleting data below " + str(new) + " [Hz].")
            self._minBand = new
            self.data = self.data[int(np.where(self.bands==new)[-1]):]
        else:
            adjNew = self.bands[int(np.where(self.bands<=new)[-1])]
            print("'" + str(new) + "' is not a valid band. " +
                  "Taking the closest band: " + str(adjNew) + " [Hz].")
            self._minBand = adjNew
            self.data = self.data[int(np.where(self.bands==adjNew)[-1]):]
        return

    @property
    def maxBand(self):
        """maximum octave fraction band.

        When a new limit is set data is automatic adjusted.

        Return:
        -------

            float.
        """
        return self._maxBand

    @maxBand.setter
    def maxBand(self, new):
        if type(new) is not int and type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        if new in self.bands:
            print("ATTENTION! Deleting data above " + str(new) + " [Hz].")
            self._maxBand = new
            self.data = self.data[:int(np.where(self.bands==new)[-1])+1]
        else:
            adjNew = self.bands[int(np.where(self.bands<=new)[-1])]
            print("'" + str(new) + "' is not a valid band. " +
                  "Taking the closest band: " + str(adjNew) + " [Hz].")
            self._maxBand = adjNew
            self.data = self.data[:int(np.where(self.bands==adjNew)[-1])+1]
        return
        # self._bandMax = new
        # return

    @property
    def range(self):
        return (self.minBand, self.maxBand)

    @property
    def data(self):
        """Fractional octave bands data.

        data must be a list or NumPy ndarray with the same number of elements
        than bands between the specified minimum (minBand) and maximum band
        (maxBand).

        Return:
        -------

            NumPy ndarray.
        """
        return self._data

    @data.setter
    def data(self, newData):
        bands = FOF(nthOct=self.nthOct,
                    freqRange=(self.minBand, self.maxBand))[:,1]
        self._minBand = float(bands[0])
        self._maxBand = float(bands[-1])
        if not isinstance(newData, list) and \
            not isinstance(newData, np.ndarray):
            raise TypeError("'data' must be provided as a list or " +
                            "numpy ndarray.")
        elif len(newData) != len(bands):
            raise ValueError("Provided 'data' has different number of bands " +
                             "then the existent bands between " +
                             "{} and {} [Hz].".format(self.minBand,
                                                      self.maxBand))

        # ...
        self._data = np.array(newData)
        self._bands = bands
        return

    @property
    def error(self):
        """error per octave fraction band.

        The error must be a list or NumPy ndarray with same number of elements
        as bands between the specified minimum (minBand) and maximum bands
        (maxBand);

        Shown as +-error.

        Return:
        -------

            NumPy ndarray.
        """
        return self._error

    @error.setter
    def error(self, newError):
        if not isinstance(newError, np.ndarray) and \
            not isinstance(newError, list) and \
                newError is not None:
            raise TypeError("'error' must be provided as a list, numpy " +
                            "ndarray or None.")
        if newError is not None:
            if len(newError) != len(self.data):
                raise ValueError("'error' must have the same length as 'data'.")
            self._error = np.array(newError)
        else:
            self._error = newError
        return

    @property
    def dataLabel(self):
        """Label of the data.

        Used for plot purposes.

        Return:
        -------

            str.
        """
        return self._dataLabel

    @dataLabel.setter
    def dataLabel(self, newLabel):
        if newLabel is not None and not isinstance(newLabel, str):
            raise TypeError("'dataLabel' must be a string or None.")
        self._dataLabel = newLabel
        return

    @property
    def errorLabel(self):
        """Label of the error information.

        Used for plot purposes.

        Return:
        -------

            str.
        """
        return self._errorLabel

    @errorLabel.setter
    def errorLabel(self, newLabel):
        if newLabel is not None and not isinstance(newLabel, str):
            raise TypeError("'errorLabel' must be a string or None.")
        self._errorLabel = newLabel
        return

    @property
    def bands(self):
        """The octave fraction bands central frequencies.

        Return:
        -------

            list with the fractional octave bands of this Analysis.
        """
        return self._bands

    # Methods

    def _h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already opened file via
        pytta.save(...).
        """
        h5group.attrs['class'] = 'Analysis'
        h5group.attrs['anType'] = self.anType
        h5group.attrs['nthOct'] = self.nthOct
        h5group.attrs['minBand'] = self.minBand
        h5group.attrs['maxBand'] = self.maxBand
        h5group.attrs['dataLabel'] = _h5.attr_parser(self.dataLabel)
        h5group.attrs['errorLabel'] = _h5.attr_parser(self.errorLabel)
        h5group.attrs['comment'] = _h5.attr_parser(self.comment)
        h5group.attrs['xLabel'] = _h5.attr_parser(self.xLabel)
        h5group.attrs['yLabel'] = _h5.attr_parser(self.yLabel)
        h5group.attrs['title'] = _h5.attr_parser(self.title)
        h5group['data'] = self.data
        if self.error is not None:
            h5group['error'] = self.error
        return

    def plot(self, **kwargs):
        """Plot the analysis data in fractinal octave bands.

        Parameters (default), (type):
        -----------------------------

            * dataLabel ('Analysis type [unit]'), (str):
                legend label for the current data

            * errorLabel ('Error'), (str):
                legend label for the current data error

            * xLabel ('Time [s]'), (str):
                x axis label.

            * yLabel ('Amplitude'), (str):
                y axis label.

            * yLim (), (list):
                inferior and superior limits.

                >>> yLim = [-100, 100]

            * title (), (str):
                plot title

            * decimalSep (','), (str):
                may be dot or comma.

                >>> decimalSep = ',' # in Brazil

            * barWidth (0.75), float:
                width of the bars from one fractional octave band.
                0 < barWidth < 1.

            * errorStyle ('standard'), str:
                error curve style. May be 'laza' or None/'standard'.

            * forceZeroCentering ('False'), bool:
                force centered bars at Y zero.

        Return:
        --------

            matplotlib.figure.Figure object.
        """
        return self.plot_bars(**kwargs)

    def plot_bars(self, dataLabel:str=None, errorLabel:str=None,
                  xLabel:str=None, yLabel:str=None,
                  yLim:list=None, xLim:list=None, title:str=None, decimalSep:str=',',
                  barWidth:float=0.75, errorStyle:str=None,
                  forceZeroCentering:bool=False, overlapBars:bool=False,
                  color:list=None):
        """Plot the analysis data in fractinal octave bands.

        Parameters (default), (type):
        -----------------------------

            * dataLabel ('Analysis type [unit]'), (str):
                legend label for the current data

            * errorLabel ('Error'), (str):
                legend label for the current data error

            * xLabel ('Time [s]'), (str):
                x axis label.

            * yLabel ('Amplitude'), (str):
                y axis label.

            * yLim (), (list):
                inferior and superior limits.

                >>> yLim = [-100, 100]

            * xLim (), (list):
                bands limits.

                >>> xLim = [100, 10000]

            * title (), (str):
                plot title

            * decimalSep (','), (str):
                may be dot or comma.

                >>> decimalSep = ',' # in Brazil

            * barWidth (0.75), float:
                width of the bars from one fractional octave band.
                0 < barWidth < 1.

            * errorStyle ('standard'), str:
                error curve style. May be 'laza' or None/'standard'.

            * forceZeroCentering ('False'), bool:
                force centered bars at Y zero.

            * overlapBars ('False'), bool:
                overlap bars. No side by side bars of different data.

            * color (None), list:
                list containing the color of each Analysis.


        Return:
        --------

            matplotlib.figure.Figure object.
        """
        if dataLabel is not None:
            self.dataLabel = dataLabel

        if errorLabel is not None:
            self.errorLabel = errorLabel


        if xLabel is not None:
            self.barsXLabel = xLabel
        else:
            if hasattr(self, 'barsXLabel'):
                if self.barsXLabel is not None:
                    xLabel = self.barsXLabel

        if yLabel is not None:
            self.barsYLabel = yLabel
        else:
            if hasattr(self, 'barsYLabel'):
                if self.barsYLabel is not None:
                    yLabel = self.barsYLabel

        if title is not None:
            self.barsTitle = title
        else:
            if hasattr(self, 'barsTitle'):
                if self.barsTitle is not None:
                    title = self.barsTitle

        fig = plot.bars((self,), xLabel, yLabel, yLim, xLim,
                        self.title, decimalSep, barWidth, errorStyle,
                        forceZeroCentering, overlapBars, color)
        return fig


class RoomAnalysis(Analysis):
    """Room monoaural acoustical parameters for quality analysis.
    
    Provides interface to estimate several room parameters based on
    the energy distribution of the impulse response. Calculations compliant to 
    ISO 3382-1 to obtain room acoustic parameters.

    It has an implementation of Lundeby et al. [1] algorithm
    to estimate the correction factor for the cumulative integral, as suggested
    by the ISO 3382-1.
    
    This class receives an one channel SignalObj or ImpulsiveResponse and 
    calculate all the room acoustic parameters.
    
    Available room parameters: D50, C80, Ts, STearly, STlate, EDT, T20, T30.
    
    Creation parameters (default), (type):
    ------------
    
        * signalArray (ndarray | list), (NumPy array):
            signal at specified domain. For 'freq' domain only half of the
            spectra must be provided. The total numSamples should also
            be provided.
            
        * ir (), (SignalObj):
            Monaural room impulse response signal.
            
        * nthOct (1), (int):
            Number of bands per octave. The default is 1.
            
        * minFreq (20), (float):
            Central frequency of the first band. The default is 2e1.
            
        * maxFreq (20000) (float):
            Central frequency of the last band. The default is 2e4.
            
        * *args : () (Tuple):
            See Analysis class.
            
        * bypassLundeby (false), (bool):
            Bypass Lundeby calculation, or not. The default is False.
            
        * suppressWarnings (false), (bool):
            Supress Lundeby warnings. The default is True.
            
        * ircut (None), (float):
            Cut the IR and throw away the silence tail. The default is None.
            
        * **kwargs (), (Dict):
            See Analysis.
            
    Attributes (default), (data type):
    -----------------------------------
        
        * parameters (), (Tuple):
        List of parameters names.
        return tuple(self._params.keys())

        * rms (), (np.ndarray):
            Effective IR amplitude by frequency `band`.

        * SPL (), (np.ndarray):
            Equivalent IR level by frequency `band`.

        * D50 (), (np.ndarray):
            Room Definition by frequency `band`.

        * C80 (), (np.ndarray):
            Room Clarity by frequency `band`.

        * Ts (), (np.ndarray):
            Central Time by frequency `band`.

        * STearly (), (np.ndarray):
            Early energy distribution by frequency `band`.

        * STlate (), (np.ndarray):
            Late energy distribution by frequency `band`.

        * EDT (), (np.ndarray):
            Early Decay Time by frequency `band`.

        * T20 (), (np.ndarray):
            Reverberation time with 20 dB decay, by frequency `band`.

        * T30 (), (np.ndarray):
            Reverberation time with 30 dB decay, by frequency `band`.
            
    Methods:
    ---------
    
        * plot_param(name [str], **kwargs):
            Plot a chart with the parameter passed in as `name`.

        * plot_rms(label [str], **kwargs):
            Plot a chart for the impulse response's `rms` by frequency `bands`.

        * plot_SPL(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `SPL` by frequency `bands`.

        * plot_C80(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `C80` by frequency `bands`.

        * plot_D50(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `D50` by frequency `bands`.

        * plot_T20(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `T20` by frequency `bands`.

        * plot_T30(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `T30` by frequency `bands`.

        * plot_Ts(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `Ts` by frequency `bands`.

        * plot_EDT(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `EDT` by frequency `bands`.

        * plot_STearly(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `STearly` by frequency `bands`.

        * plot_STlate(label [str], yaxis [str], **kwargs):
            Plot a chart for the impulse response's `STlate` by frequency `bands`.
    
    For further information on methods see its specific documentation.
    
    Authors:
        João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
        Matheus Lazarin, matheus.lazarin@eac.ufsm.br
        Rinaldi Petrolli, rinaldi.petrolli@eac.ufsm.br"""

    def __init__(self, ir: SignalObj, nthOct: int = 1,
                 minFreq: float = 2e1, maxFreq: float = 2e4, *args,
                 plotLundeby: bool = False,
                 bypassLundeby: bool = False,
                 suppressWarnings: bool = True,
                 ircut: float = None, **kwargs):
        _ir = ir.IR if type(ir) == ImpulsiveResponse else ir
        minBand = freq_to_band(minFreq, nthOct, 1000, 10)
        maxBand = freq_to_band(maxFreq, nthOct, 1000, 10)
        nbands = maxBand - minBand + 1
        super().__init__('mixed', nthOct, minFreq, maxFreq, nbands*[0], *args, **kwargs)
        self.ir = crop_IR(_ir, ircut)
        self._params = self.estimate_energy_parameters(self.ir, self.bands, plotLundeby,
                                                       bypassLundeby, suppressWarnings,
                                                       nthOct=nthOct, minFreq=minFreq,
                                                       maxFreq=maxFreq)
        return

    @staticmethod
    def estimate_energy_parameters(ir: SignalObj, bands: np.ndarray,
                                   plotLundeby: bool = False,
                                   bypassLundeby: bool = False,
                                   suppressWarnings: bool = False, **kwargs):
        """
        Estimate the Impulse Response energy parameters.

        Parameters
        ----------
        bypassLundeby : bool
            Whether to bypass calculation of Lundeby IR improvements or not.
            The default is False.
        suppressWarnings : bool
            If supress warnings about IR quality and the bypassing of Lundeby calculations.
            The default is False.

        Returns
        -------
        params : Dict[str, np.ndarray]
            A dict with parameters by name.

        """
        listEDC, fhSignal = cumulative_integration(ir, bypassLundeby, plotLundeby, suppressWarnings, **kwargs)
        params = {}
        params['rms'] = fhSignal.rms()
        params['SPL'] = fhSignal.spl()
        params['Ts'] = central_time(fhSignal.timeSignal**2, fhSignal.timeVector)
        params['D50'] = definition(listEDC, ir.samplingRate)
        params['C80'] = clarity(listEDC, ir.samplingRate)
        params['STearly'] = st_early(listEDC, ir.samplingRate)
        params['STlate'] = st_late(listEDC, ir.samplingRate)
        params['EDT'] = reverberation_time('EDT', listEDC)
        params['T20'] = reverberation_time(20, listEDC)
        params['T30'] = reverberation_time(30, listEDC)
        # self._params['BR'], self._params['TR'] = timbre_ratios(self.T20)
        return params

    @property
    def parameters(self):
        """List of parameters names."""
        return tuple(self._params.keys())

    @property
    def rms(self):
        """Effective IR amplitude by frequency `band`."""
        return self._params['rms']

    @property
    def SPL(self):
        """Equivalent IR level by frequency `band`."""
        return self._params['SPL']

    @property
    def D50(self):
        """Room Definition by frequency `band`."""
        return self._params['D50']

    @property
    def C80(self):
        """Effective IR amplitude, by frequency `band`."""
        return self._params['C80']

    @property
    def Ts(self):
        """Central Time by frequency `band`."""
        return self._params['Ts']

    @property
    def STearly(self):
        """Early energy distribution by frequency `band`."""
        return self._params['STearly']

    @property
    def STlate(self):
        """Late energy distribution by frequency `band`."""
        return self._params['STlate']

    @property
    def EDT(self):
        """Early Decay Time by frequency `band`."""
        return self._params['EDT']

    @property
    def T20(self):
        """Reverberation time with 20 dB decay, by frequency `band`."""
        return self._params['T20']

    @property
    def T30(self):
        """Reverberation time with 30 dB decay, by frequency `band`."""
        return self._params['T30']

    # @property
    # def BR(self):
    #     """Reverberation time with 30 dB decay, by frequency `band`."""
    #     return self._params['BR']

    # @property
    # def TR(self):
    #     """Reverberation time with 30 dB decay, by frequency `band`."""
    #     return self._params['TR']

    def plot_param(self, name: str, **kwargs):
        """
        Plot a chart with the parameter passed in as `name`.


        Parameters
        ----------
        name : str
            Room parameter name, e.g. `'T20' | 'C80' | 'SPL'`, etc.
        kwargs: Dict
            All kwargs accepted by `Analysis.plot_bar`.

        Returns
        -------
        f : matplotlib.Figure
            The figure of the plot chart.

        """
        self._data = self._params[name]
        f = self.plot(**kwargs)
        self._data = np.zeros(self.bands.shape)
        return f

    def plot_rms(self, label='RMS', **kwargs):
        """Plot a chart for the impulse response's `rms` by frequency `bands`."""
        return self.plot_param('rms', dataLabel=label, **kwargs)

    def plot_SPL(self, label='SPL', yaxis='Level [dB]', **kwargs):
        """Plot a chart for the impulse response's `SPL` by frequency `bands`."""
        return self.plot_param('SPL', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_C80(self, label='C80', yaxis='Clarity [dB]', **kwargs):
        """Plot a chart for the impulse response's `C80` by frequency `bands`."""
        return self.plot_param('C80', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_D50(self, label='D50', yaxis='Definition [%]', **kwargs):
        """Plot a chart for the impulse response's `D50` by frequency `bands`."""
        return self.plot_param('D50', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_T20(self, label='T20', yaxis='Reverberation time [s]', **kwargs):
        """Plot a chart for the impulse response's `T20` by frequency `bands`."""
        return self.plot_param('T20', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_T30(self, label='T30', yaxis='Reverberation time [s]', **kwargs):
        """Plot a chart for the impulse response's `T30` by frequency `bands`."""
        return self.plot_param('T30', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_Ts(self, label='Ts', yaxis='Central time [s]', **kwargs):
        """Plot a chart for the impulse response's `Ts` by frequency `bands`."""
        return self.plot_param('Ts', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_EDT(self, label='EDT', yaxis='Early Decay Time [s]', **kwargs):
        """Plot a chart for the impulse response's `EDT` by frequency `bands`."""
        return self.plot_param('EDT', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_STearly(self, label='STearly', yaxis='Early reflection level [dB]', **kwargs):
        """Plot a chart for the impulse response's `STearly` by frequency `bands`."""
        return self.plot_param('STearly', dataLabel=label, yLabel=yaxis, **kwargs)

    def plot_STlate(self, label='STlate', yaxis='Late reflection level [dB]', **kwargs):
        """Plot a chart for the impulse response's `STlate` by frequency `bands`."""
        return self.plot_param('STlate', dataLabel=label, yLabel=yaxis, **kwargs)

    # def plot_BR(self):
    #     """Plot a chart for the impulse response's `BR` by frequency `bands`."""
    #     return self.plot_param('BR')

    # def plot_TR(self):
    #     """Plot a chart for the impulse response's `TR` by frequency `bands`."""
    #     return self.plot_param('TR')


def _filter(signal,
            order: int = 4,
            nthOct: int = 3,
            minFreq: float = 20,
            maxFreq: float = 20000,
            refFreq: float = 1000,
            base: int = 10):
    of = OctFilter(order=order,
                   nthOct=nthOct,
                   samplingRate=signal.samplingRate,
                   minFreq=minFreq,
                   maxFreq=maxFreq,
                   refFreq=refFreq,
                   base=base)
    result = of.filter(signal)
    return result[0]


# @njit
def _level_profile(timeSignal, samplingRate,
                   numSamples, numChannels, blockSamples=None):
    """Get h(t) in octave bands and do the local time averaging in nblocks. Returns h^2_averaged(block)."""
    def mean_squared(x):
        return np.mean(x**2)

    if blockSamples is None:
        blockSamples = 100
    nblocks = int(numSamples // blockSamples)
    profile = np.zeros((nblocks, numChannels), dtype=np.float32)
    timeStamp = np.zeros((nblocks, 1))

    for ch in range(numChannels):
        # if numChannels == 1:
        #     tmp = timeSignal
        # else:
        tmp = timeSignal[:, ch]
        for idx in range(nblocks):
            profile[idx, ch] = mean_squared(tmp[:blockSamples])
            timeStamp[idx, 0] = idx*blockSamples/samplingRate
            tmp = tmp[blockSamples:]
    return profile, timeStamp


# @njit
def _start_sample_ISO3382(timeSignal, threshold) -> np.ndarray:
    squaredIR = timeSignal**2
    # assume the last 10% of the IR is noise, and calculate its noise level
    last10Idx = -int(len(squaredIR)//10)
    noiseLevel = np.mean(squaredIR[last10Idx:])
    # get the maximum of the signal, that is the assumed IR peak
    max_val = np.max(squaredIR)
    max_idx = np.argmax(squaredIR)
    # check if the SNR is enough to assume that the signal is an IR. If not,
    # the signal is probably not an IR, so it starts at sample 1
    idxNoShift = np.asarray([max_val < 100*noiseLevel or
                             max_idx > int(0.9*squaredIR.shape[0])])
    # less than 20dB SNR or in the "noisy" part
    if idxNoShift.any():
        print("noiseLevelCheck: The SNR too bad or this is not an " +
              "impulse response.")
        return 0
    # find the first sample that lies under the given threshold
    threshold = abs(threshold)
    startSample = 1
#    # TODO - envelope mar/pdi - check!
#    if idxNoShift:
#        print("Something wrong!")
#        return
    # if maximum lies on the first point, then there is no point in searching
    # for the beginning of the IR. Just return this position.
    if max_idx > 0:
        abs_dat = 10*np.log10(squaredIR[:max_idx]) \
                  - 10.*np.log10(max_val)
        thresholdNotOk = True
        thresholdShift = 0
        while thresholdNotOk:
            if len(np.where(abs_dat < (-threshold+thresholdShift))[0]) > 0:
                lastBelowThreshold = \
                    np.where(abs_dat < (-threshold+thresholdShift))[0][-1]
                thresholdNotOk = False
            else:
                thresholdShift += 1
        if thresholdShift > 0:
            print("_start_sample_ISO3382: 20 dB threshold too high. " +
                  "Decreasing it.")
        if lastBelowThreshold > 0:
            startSample = lastBelowThreshold
        else:
            startSample = 1
    return startSample


# @njit
def _circular_time_shift(timeSignal, threshold=20):
    # find the first sample where inputSignal level > 20 dB or > bgNoise level
    startSample = _start_sample_ISO3382(timeSignal, threshold)
    newTimeSignal = timeSignal[startSample:]
    return (newTimeSignal, startSample)


# @njit
def _Lundeby_correction(band, timeSignal, samplingRate, numSamples,
                        numChannels, timeLength, suppressWarnings=True):
    returnTuple = (np.float32(0), np.float32(0), np.int32(0), np.float32(0))
    timeSignal, sampleShift = _circular_time_shift(timeSignal)
    if sampleShift is None:
        return returnTuple

    numSamples -= sampleShift  # discount shifted samples
    numParts = 5  # number of parts per 10 dB decay. N = any([3, 10])
    dBtoNoise = 7  # stop point 10 dB above first estimated background noise
    useDynRange = 15  # dynamic range

    # Window length - 10 to 50 ms, longer periods for lower frequencies and vice versa
    repeat = True
    i = 0
    winTimeLength = 0.01
    while repeat: # loop to find proper winTimeLength
        winTimeLength = winTimeLength + 0.01*i
        # 1) local time average:
        blockSamples = int(winTimeLength * samplingRate)
        timeWinData, timeVecWin = _level_profile(timeSignal, samplingRate,
                                                  numSamples, numChannels,
                                                  blockSamples)

        # 2) estimate noise from h^2_averaged(block):
        bgNoiseLevel = 10 * \
                       np.log10(
                                np.mean(timeWinData[-int(timeWinData.size/10):]))

        # 3) Calculate preliminar slope
        startIdx = np.argmax(np.abs(timeWinData/np.max(np.abs(timeWinData))))
        stopIdx = startIdx + np.where(10*np.log10(timeWinData[startIdx+1:])
                                      >= bgNoiseLevel + dBtoNoise)[0][-1]
        dynRange = 10*np.log10(timeWinData[stopIdx]) \
            - 10*np.log10(timeWinData[startIdx])
        if (stopIdx == startIdx) or (dynRange > -5)[0]:
            if not suppressWarnings:
                print(band, "[Hz] band: SNR too low for the preliminar slope",
                  "calculation.")
            # return returnTuple

        # X*c = EDC (energy decaying curve)
        X = np.ones((stopIdx-startIdx, 2), dtype=np.float32)
        X[:, 1] = timeVecWin[startIdx:stopIdx, 0]
        c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]),
                            rcond=-1)[0]

        if (c[1] == 0)[0] or np.isnan(c).any():
            if not suppressWarnings:
                print(band, "[Hz] band: regression failed. T would be inf.")
            # return returnTuple

        # 4) preliminary intersection
        crossingPoint = (bgNoiseLevel - c[0]) / c[1]  # [s]
        if (crossingPoint > 2*(timeLength + sampleShift/samplingRate))[0]:
            if not suppressWarnings:
                print(band, "[Hz] band: preliminary intersection point between",
                      "bgNoiseLevel and the decay slope greater than signal length.")
            # return returnTuple

        # 5) new local time interval length
        nBlocksInDecay = numParts * dynRange[0] / -10

        dynRangeTime = timeVecWin[stopIdx] - timeVecWin[startIdx]
        blockSamples = int(samplingRate * dynRangeTime[0] / nBlocksInDecay)

        # 6) average
        timeWinData, timeVecWin = _level_profile(timeSignal, samplingRate,
                                                 numSamples, numChannels,
                                                 blockSamples)

        oldCrossingPoint = 11+crossingPoint  # arbitrary higher value to enter loop
        loopCounter = 0

        while (np.abs(oldCrossingPoint - crossingPoint) > 0.001)[0]:
            # 7) estimate background noise level (BGL)
            bgNoiseMargin = 7
            idxLast10Percent = int(len(timeWinData)-(len(timeWinData)//10))
            bgStartTime = crossingPoint - bgNoiseMargin/c[1]
            if (bgStartTime > timeVecWin[-1:][0])[0]:
                idx10dBDecayBelowCrossPoint = len(timeVecWin)-1
            else:
                idx10dBDecayBelowCrossPoint = \
                    np.where(timeVecWin >= bgStartTime)[0][0]
            BGL = np.mean(timeWinData[np.min(
                    np.array([idxLast10Percent,
                              idx10dBDecayBelowCrossPoint])):])
            bgNoiseLevel = 10*np.log10(BGL)

            # 8) estimate late decay slope
            stopTime = (bgNoiseLevel + dBtoNoise - c[0])/c[1]
            if (stopTime > timeVecWin[-1])[0]:
                stopIdx = 0
            else:
                stopIdx = int(np.where(timeVecWin >= stopTime)[0][0])

            startTime = (bgNoiseLevel + dBtoNoise + useDynRange - c[0])/c[1]
            if (startTime < timeVecWin[0])[0]:
                startIdx = 0
            else:
                startIdx = int(np.where(timeVecWin <= startTime)[0][0])

            lateDynRange = np.abs(10*np.log10(timeWinData[stopIdx]) \
                - 10*np.log10(timeWinData[startIdx]))

            # where returns empty
            if stopIdx == startIdx or (lateDynRange < useDynRange)[0]:
                if not suppressWarnings:
                    print(band, "[Hz] band: SNR for the Lundeby late decay slope too",
                        "low. Skipping!")
                # c[1] = np.inf
                c[1] = 0
                i += 1
                break

            X = np.ones((stopIdx-startIdx, 2), dtype=np.float32)
            X[:, 1] = timeVecWin[startIdx:stopIdx, 0]
            c = np.linalg.lstsq(X, 10*np.log10(timeWinData[startIdx:stopIdx]),
                                rcond=-1)[0]

            if (c[1] >= 0)[0]:
                if not suppressWarnings:
                    print(band, "[Hz] band: regression did not work, T -> inf.",
                        "Setting slope to 0!")
                # c[1] = np.inf
                c[1] = 0
                i += 1
                break

            # 9) find crosspoint
            oldCrossingPoint = crossingPoint
            crossingPoint = (bgNoiseLevel - c[0]) / c[1]

            loopCounter += 1
            if loopCounter > 30:
                if not suppressWarnings:
                    print(band, "[Hz] band: more than 30 iterations on regression.",
                        "Canceling!")
                break

        interIdx = crossingPoint * samplingRate # [sample]
        i += i
        if c[1][0] != 0:
            repeat = False
        if i > 5:
            if not suppressWarnings:
                print(band, "[Hz] band: too many iterations to find winTimeLength.", "Canceling!")
            return returnTuple

    return c[0][0], c[1][0], np.int32(interIdx[0]), BGL


# @njit
def energy_decay_calculation(band, timeSignal, timeVector, samplingRate,
                             numSamples, numChannels, timeLength, bypassLundeby,
                             suppressWarnings=True):
    """Calculate the Energy Decay Curve."""
    if bypassLundeby is False:
        lundebyParams = \
            _Lundeby_correction(band,
                                timeSignal,
                                samplingRate,
                                numSamples,
                                numChannels,
                                timeLength,
                                suppressWarnings=suppressWarnings)
        _, c1, interIdx, BGL = lundebyParams
        lateRT = -60/c1 if c1 != 0 else 0
    else:
        interIdx = 0
        lateRT = 1

    if interIdx == 0:
        interIdx = -1

    truncatedTimeSignal = timeSignal[:interIdx, 0]
    truncatedTimeVector = timeVector[:interIdx]

    if lateRT != 0.0:
        if not bypassLundeby:
            C = samplingRate*BGL*lateRT/(6*np.log(10))
        else:
            C = 0
        sqrInv = truncatedTimeSignal[::-1]**2
        energyDecayFull = np.cumsum(sqrInv)[::-1] + C
        energyDecay = energyDecayFull/energyDecayFull[0]
    else:
        if not suppressWarnings:
            print(band, "[Hz] band: could not estimate C factor")
        C = 0
        energyDecay = np.zeros(truncatedTimeVector.size)
    return (energyDecay, truncatedTimeVector, lundebyParams)


def cumulative_integration(inputSignal,
                           bypassLundeby,
                           plotLundebyResults,
                           suppressWarnings=True,
                           **kwargs):
    """Cumulative integration with proper corrections."""

    def plot_lundeby():
        c0, c1, interIdx, BGL = lundebyParams
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_axes([0.08, 0.15, 0.75, 0.8], polar=False,
                          projection='rectilinear', xscale='linear')
        line = c1*inputSignal.timeVector + c0
        ax.plot(inputSignal.timeVector, 10*np.log10(inputSignal.timeSignal**2), label='IR')
        ax.axhline(y=10*np.log10(BGL), color='#1f77b4', label='BG Noise', c='red')
        ax.plot(inputSignal.timeVector, line, label='Late slope', c='black')
        ax.axvline(x=interIdx/inputSignal.samplingRate, label='Truncation point', c='green')
        ax.grid()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude [dBFS]')
        plt.title('{0:.0f} [Hz]'.format(band))
        ax.legend(loc='best', shadow=True, fontsize='x-large')

    # timeSignal = inputSignal.timeSignal[:]
    # Substituted by SignalObj.crop in analyse function
    # timeSignal, sampleShift = _circular_time_shift(timeSignal)
    # del sampleShift
    # hSignal = SignalObj(timeSignal, inputSignal.lengthDomain, inputSignal.samplingRate)
    hSignal = _filter(inputSignal, **kwargs)
    bands = FOF(nthOct=kwargs['nthOct'],
                freqRange=[kwargs['minFreq'],
                           kwargs['maxFreq']])[:, 1]
    listEDC = []
    for ch in range(hSignal.numChannels):
        signal = hSignal[ch]
        band = bands[ch]
        energyDecay, energyVector, lundebyParams = \
            energy_decay_calculation(band,
                                     signal.timeSignal,
                                     signal.timeVector,
                                     signal.samplingRate,
                                     signal.numSamples,
                                     signal.numChannels,
                                     signal.timeLength,
                                     bypassLundeby,
                                     suppressWarnings=suppressWarnings)
        listEDC.append((energyDecay, energyVector))
        if plotLundebyResults:  # Placed here because Numba can't handle plots.
            # plot_lundeby(band, timeVector, timeSignal,  samplingRate,
            #             lundebyParams)
            plot_lundeby()
    return listEDC, hSignal

# @njit
def reverb_time_regression(energyDecay, energyVector, upperLim, lowerLim):
    """Interpolate the EDT to get the reverberation time."""
    if not np.any(energyDecay):
        return 0
    first = np.where(10*np.log10(energyDecay) >= upperLim)[0][-1]
    last = np.where(10*np.log10(energyDecay) >= lowerLim)[0][-1]
    if last <= first:
        # return np.nan
        return 0
    X = np.ones((last-first, 2))
    X[:, 1] = energyVector[first:last]
    c = np.linalg.lstsq(X, 10*np.log10(energyDecay[first:last]), rcond=-1)[0]
    return -60/c[1]


def reverberation_time(decay, listEDC):
    """Call the reverberation time regression."""
    try:
        decay = int(decay)
        y1 = -5
        y2 = y1 - decay
    except ValueError:
        if decay in ['EDT', 'edt']:
            y1 = 0
            y2 = -10
        else:
            raise ValueError("Decay must be either 'EDT' or an integer \
                             corresponding to the amount of energy decayed to \
                             evaluate, e.g. (decay='20' | 20).")
    RT = []
    for ED in listEDC:
        edc, edv = ED
        RT.append(reverb_time_regression(edc, edv, y1, y2))
    return np.array(RT, dtype='float32')


def definition(listEDC: list, fs: int, t: int = 50) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.
    t_ms : int, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    definition : np.ndarray
        The room "Definition" parameter, in percentage [%].

    """
    t_ms = t * fs // 1000
    definition = np.zeros((len(listEDC), ), dtype='float32')
    for band, pair in enumerate(listEDC):
        int_h2 = pair[0][0]  # sum of squared IR from start to the end
        intr_h2_ms = pair[0][t_ms]  # sum of squared IR from the interval to the end
        int_h2_ms = int_h2 - intr_h2_ms  # sum of squared IR from start to interval
        definition[band] = (int_h2_ms / int_h2)
    # sumSIRt = sqrIR.sum(axis=0)  # total sum of squared IR
    # sumSIRi = sqrIR[:t_ms].sum(axis=0)  # sum of initial portion of squared IR
    # definition = np.round(100 * (sumSIRi / sumSIRt), 2)  # [%]
    return np.round(100 * definition, 2)  # [%]


def clarity(listEDC: list, fs: int, t: int = 80) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.
    t_ms : int, optional
        DESCRIPTION. The default is 80.

    Returns
    -------
    clarity : np.ndarray
        The room "Clarity" parameter, in decibel [dB].

    """
    t_ms = t * fs // 1000
    clarity = np.zeros((len(listEDC), ), dtype='float32')
    for band, pair in enumerate(listEDC):
        int_h2 = pair[0][0]  # sum of squared IR from start to the end
        intr_h2_ms = pair[0][t_ms]  # sum of squared IR from the interval to the end
        int_h2_ms = int_h2 - intr_h2_ms  # sum of squared IR from start to interval
        clarity[band] = 10 * np.log10(int_h2_ms / intr_h2_ms)  # [dB]
    # sumSIRi = sqrIR[:t_ms].sum(axis=0)  # sum of initial portion of squared IR
    # sumSIRe = sqrIR[t_ms:].sum(axis=0)  # sum of ending portion of squared IR
    # clarity = np.round(10 * np.log10(sumSIRi / sumSIRe), 2)  # [dB]
    return np.round(clarity, 2)


def central_time(sqrIR: np.ndarray, tstamp: np.ndarray) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        Squared room impulsive response.
    tstamp : np.ndarray
        Time stamps of each IR sample.

    Returns
    -------
    central_time : np.ndarray
        The time instant that balance of energy is equal before and after it.

    """
    sumSIR = sqrIR.sum(axis=0)
    sumTSIR = (tstamp[:, None] * sqrIR).sum(axis=0)
    central_time = (sumTSIR / sumSIR) * 1000  # milisseconds
    return central_time


def st_early(listEDC: list, fs: int) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.

    Returns
    -------
    STearly : np.ndarray
        DESCRIPTION.

    """
    ms = fs // 1000
    STearly = np.zeros((len(listEDC), ), dtype='float32')
    for band, pair in enumerate(listEDC):
        int_h2 = pair[0][0]  # sum of squared IR from start to the end
        intr_h2_10ms = pair[0][10 * ms]  # sum of squared IR from the interval to the end
        int_h2_10ms = int_h2 - intr_h2_10ms

        intr_h2_20ms = pair[0][20 * ms]  # sum of squared IR from the interval to the end
        intr_h2_100ms = pair[0][100 * ms]  # sum of squared IR from the interval to the end
        int_h2_20a100ms = intr_h2_20ms - intr_h2_100ms
        STearly[band] = 10 * np.log10(int_h2_20a100ms / int_h2_10ms)  # [dB]

    # sum10ms = sqrIR[:int(10 * ms)].sum(axis=0)
    # sum20ms = sqrIR[int(20 * ms):int(100 * ms)].sum(axis=0)
    # STearly = 10 * np.log10(sum20ms / sum10ms)
    return np.round(STearly, 4)


def st_late(listEDC: list, fs: int) -> np.ndarray:
    """
    Room parameter.

    Parameters
    ----------
    sqrIR : np.ndarray
        DESCRIPTION.

    Returns
    -------
    STlate : np.ndarray
        DESCRIPTION.

    """
    ms = fs // 1000
    STlate = np.zeros((len(listEDC), ), dtype='float32')
    for band, pair in enumerate(listEDC):
        int_h2 = pair[0][0]  # sum of squared IR from start to the end
        intr_h2_10ms = pair[0][10 * ms]  # sum of squared IR from the interval to the end
        int_h2_10ms = int_h2 - intr_h2_10ms  # sum of squared IR from start to interval
        intr_h2_100ms = pair[0][100 * ms]  # sum of squared IR from the interval to the end
        STlate[band] = 10 * np.log10(intr_h2_100ms / int_h2_10ms)  # [dB]
    # sum10ms = sqrIR[:int(10 * ms)].sum(axis=0)
    # sum100ms = sqrIR[int(100 * ms):int(1000 * ms)].sum(axis=0)
    # STlate = 10 * np.log10(sum100ms / sum10ms)
    return np.round(STlate, 4)


def crop_IR(SigObj, IREndManualCut):
    """Cut the impulse response at background noise level."""
    timeSignal = cp.copy(SigObj.timeSignal)
    timeVector = SigObj.timeVector
    samplingRate = SigObj.samplingRate
    numSamples = SigObj.numSamples
    # numChannels = SigObj.numChannels
    if SigObj.numChannels > 1:
        print('crop_IR: The provided impulsive response has more than one ' +
              'channel. Cropping based on channel 1.')
    numChannels = 1
    # Cut the end automatically or manual
    if IREndManualCut is None:
        winTimeLength = 0.1  # [s]
        meanSize = 5  # [blocks]
        dBtoReplica = 6  # [dB]
        blockSamples = int(winTimeLength * samplingRate)
        timeWinData, timeVecWin = _level_profile(timeSignal, samplingRate,
                                                numSamples, numChannels,
                                                blockSamples)
        endTimeCut = timeVector[-1]
        for blockIdx, blockAmplitude in enumerate(timeWinData):
            if blockIdx >= meanSize:
                anteriorMean = 10*np.log10( \
                    np.sum(timeWinData[blockIdx-meanSize:blockIdx])/meanSize)
                if 10*np.log10(blockAmplitude) > anteriorMean+dBtoReplica:
                    endTimeCut = timeVecWin[blockIdx-meanSize//2]
                    break
    else:
        endTimeCut = IREndManualCut
    endTimeCutIdx = np.where(timeVector >= endTimeCut)[0][0]
    timeSignal = timeSignal[:endTimeCutIdx]
    # Cut the start automatically
    timeSignal, _ = _circular_time_shift(timeSignal)
    result = SignalObj(timeSignal,
                       'time',
                       samplingRate,
                       signalType='energy')
    return result
