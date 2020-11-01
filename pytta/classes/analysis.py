# -*- coding: utf-8 -*-

from pytta.classes._instanceinfo import RememberInstanceCreationInfo as RICI
from pytta.classes.filter import fractional_octave_frequencies as FOF
from math import isnan
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import locale
from pytta import _h5utils as _h5
from pytta import _plot as plot


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
            The data with the exact number of bands between the spcified minimum
            (minBand) and maximum band (maxBand);

        * dataLabel (''), (string):
            Label for plots;

        * error, (list | numpy array):
            The error with the exact number of bands between the spcified
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
            When a new limit is setted data is automatic adjusted.

        * maxBand, (int | float):
            When a new limit is setted data is automatic adjusted.


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
            raise TypeError("Analysys can only be operated with int, float, " +
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

        When a new limit is setted data is automatic adjusted.

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
            print("ATENTION! Deleting data below " + str(new) + " [Hz].")
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

        When a new limit is setted data is automatic adjusted.

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
            print("ATENTION! Deleting data above " + str(new) + " [Hz].")
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
        than bands between the spcified minimum (minBand) and maximum band
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
                             "then the existant bands betwen " +
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
        as bands between the spcified minimum (minBand) and maximum bands
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
        Saves itself inside a hdf5 group from an already openned file via
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
