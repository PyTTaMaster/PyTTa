# -*- coding: utf-8 -*-

from pytta.classes._instanceinfo import RememberInstanceCreationInfo as RICI
from pytta.classes.filter import fractional_octave_frequencies as FOF
from math import isnan
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import locale
from pytta import h5utils as _h5


# Analysis types and its units
anTypes = {'RT': ('s', 'Reverberation time'),
           'C': ('dB', 'Clarity'),
           'D': ('%', 'Definition'),
           'G': ('dB', 'Strength factor'),
           'L': ('dB', 'Level'),
           'mixed': ('-', 'Mixed')}

class Analysis(RICI):
    """
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
        self.minBand = minBand
        self.maxBand = maxBand
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
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data)

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
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data)

        return result

    def __mul__(self, other):
        if isinstance(other, Analysis):
            if other.range != self.range:
                raise ValueError("Can't subtract! Both Analysis have " +
                                "different band limits.")
            result = Analysis(anType='mixed', nthOct=self.nthOct,
                            minBand=self.minBand, maxBand=self.maxBand,
                            data=self.data*other.data)
        elif isinstance(other, (int, float)):
            result = Analysis(anType='mixed', nthOct=self.nthOct,
                            minBand=self.minBand, maxBand=self.maxBand,
                            data=self.data*other)
        else:
            raise TypeError("Analysys can only be operated with int, float, " +
                            "or Analysis types.")
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
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data)

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
        result = Analysis(anType=anType, nthOct=self.nthOct,
                        minBand=self.minBand, maxBand=self.maxBand,
                        data=data)

        return result

    # Properties

    @property
    def anType(self):
        """
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
                             "'C' or 'D'.")
        self.unit = anTypes[newType][0]
        self.anName = anTypes[newType][1]
        self._anType = newType
        return

    @property
    def nthOct(self):
        """
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
        """
        """
        return self._bandMin

    @minBand.setter
    def minBand(self, new):
        if type(new) is not int and type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._bandMin = new
        return

    @property
    def maxBand(self):
        """
        """
        return self._bandMax

    @maxBand.setter
    def maxBand(self, new):
        if type(new) is not int and type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._bandMax = new
        return
    
    @property
    def range(self):
        return (self.minBand, self.maxBand)

    @property
    def data(self):
        """data [summary] # TO DO
        
        :return: [description]
        :rtype: [type]
        """
        return self._data

    @data.setter
    def data(self, newData):
        bands = FOF(nthOct=self.nthOct,
                    minFreq=self.minBand,
                    maxFreq=self.maxBand)[:,1]
        self.minBand = float(bands[0])
        self.maxBand = float(bands[-1])
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
        """error [summary] # TO DO
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
        """dataLabel [summary] # TO DO
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
        """errorLabel [summary] # TO DO
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
        """
        """
        return self._bands


    # Methods

    # def _to_dict(self):
    #     return

    # def pytta_save(self, dirname=time.ctime(time.time())):
    #     return

    def h5_save(self, h5group):
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
        return self.plot_bars(**kwargs)

    def plot_bars(self, xLabel=None, yLabel=None, yLim=None, errorLabel=None,
                  title=None, decimalSep=','):
        """
        Analysis bar plotting method
        """
        if decimalSep == ',':
            locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
            plt.rcParams['axes.formatter.use_locale'] = True
        elif decimalSep =='.':
            locale.setlocale(locale.LC_NUMERIC, 'C')
            plt.rcParams['axes.formatter.use_locale'] = False
        else:
            raise ValueError("'decimalSep' must be the string '.' or ','.")

        if xLabel is None:
            if self.xLabel is None:
                xLabel = 'Frequency bands [Hz]'
            else:
                xLabel = self.xLabel
        else:
            self.xLabel = xLabel
        
        if yLabel is None:
            if self.yLabel is None:
                yLabel = 'Modulus [{}]'
                yLabel = yLabel.format(self.unit)
            else:
                yLabel = self.yLabel
        else:
            self.yLabel = yLabel
        
        if title is None:
            if self.title is None:
                title = '{} analysis'
                title = title.format(self.anName)
            else:
                title = self.title
        else:
            self.title = title
        
        if errorLabel is None:
            if self.errorLabel is None:
                errorLabel = 'Error'
            else:
                errorLabel = self.errorLabel
        else:
            self.errorLabel = errorLabel

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                          projection='rectilinear', xscale='linear')
        ax.set_snap(True)
        fbar = range(len(self.data))
        

        negativeCounter = 0
        for value in self.data:
            if value < 0:
                negativeCounter += 1

        if negativeCounter > len(self.data)//2:
            minval = np.amin(self.data)
            marginData = [value for value in self.data if not np.isinf(value)]
            margin = \
                np.abs((np.nanmax(marginData) - np.nanmin(marginData)) / 20)
            minval = minval - margin
            minval += np.sign(minval)
            ax.bar(*zip(*enumerate(-minval + self.data)), width=0.75,
                   label=self.dataLabel, zorder=-1)
            if self.error is not None:
                ax.errorbar(*zip(*enumerate(-minval + self.data)),
                            yerr=self.error, fmt='none',
                            ecolor='limegreen', elinewidth=23, zorder=0,
                            fillstyle='full', alpha=.60, label=errorLabel)

        else:
            ax.bar(fbar, self.data, width=0.75, label=self.dataLabel, zorder=-1)
            if self.error is not None:
                ax.errorbar(fbar, self.data, yerr=self.error, fmt='none',
                            ecolor='limegreen', elinewidth=23, zorder=0,
                            fillstyle='full', alpha=.60, label=errorLabel)
            minval = 0

        error = self.error if self.error is not None else 0

        ylimInfData = -minval + self.data - error
        ylimInfData = [value for value in ylimInfData if not np.isinf(value)]
        ylimInfMargin = \
            np.abs((np.nanmax(ylimInfData) - np.nanmin(ylimInfData)) / 20)
        ylimInf = np.nanmin(ylimInfData) - ylimInfMargin
    
        ylimSupData = -minval + self.data + error
        ylimSupData = [value for value in ylimSupData if not np.isinf(value)]
        ylimSupMargin = \
            np.abs((np.nanmax(ylimSupData) - np.nanmin(ylimSupData)) / 20)
        ylimSup = np.nanmax(ylimSupData) + ylimSupMargin
        
        if yLim is None:
            yLim = (ylimInf, ylimSup)
        ax.set_ylim(yLim)
        
        ax.grid(color='gray', linestyle='-.', linewidth=0.4)

        ax.set_xticks(fbar)
        xticks = self.bands
        ax.set_xticklabels(['{:n}'.format(tick) for tick in xticks],
                           rotation=45, fontsize=14)
        ax.set_xlabel(xLabel, fontsize=20)

        # Adjust yticks    
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.14/5))
        # ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:#.2n}'))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)

        # yticks = np.linspace(*yLim, 11)
        # ax.set_yticks(yticks.tolist())
        # yticklabels = yticks + minval
        # ax.set_yticklabels(['{:n}'.format(float('{0:.2f}'.format(tick)))
        #                     for tick in yticklabels.tolist()], fontsize=14)

        ax.set_ylabel(yLabel, fontsize=20)
        
        ax.legend(loc='best')

        plt.title(title, fontsize=20)
        
        return fig
