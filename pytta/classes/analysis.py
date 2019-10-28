# -*- coding: utf-8 -*-

from pytta.classes.filter import fractional_octave_frequencies as FOF
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import time
import locale


# Analysis types and its units
anTypes = {'RT': ('s', 'Reverberation time'),
           'C': ('dB', 'Clarity'),
           'D': ('%', 'Definition'),
           'L': ('dB', 'Level'),
           'mixed': ('-', 'Mixed')}

class Analysis(object):
    """
    """

    # Magic methods

    def __init__(self,
                anType,
                nthOct,
                minBand,
                maxBand,
                data,
                comment='No comments.'):
        self.anType = anType
        self.nthOct = nthOct
        self.minBand = minBand
        self.maxBand = maxBand
        self.data = data
        self.comment = comment
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

    def __truediv__(self, other):
        # if isinstance(other, Analysis):
        #     if other.range != self.range:
        #         raise ValueError("Can't subtract! Both Analysis have " +
        #                         "different band limits.")
        #     result = Analysis(anType='mixed', nthOct=self.nthOct,
        #                     minBand=self.minBand, maxBand=self.maxBand,
        #                     data=self.data/other.data)
        # elif isinstance(other, (int, float)):
        #     result = Analysis(anType='mixed', nthOct=self.nthOct,
        #                     minBand=self.minBand, maxBand=self.maxBand,
        #                     data=self.data/other)
        # else:
        #     raise TypeError("Analysys can only be operated with int, float, " +
        #                     "or Analysis types.")
        # return result
        if isinstance(other, Analysis):
            if self.anType == 'L':
                if other.range != self.range:
                    raise ValueError("Can't divide! Both Analysis have" +
                                     " different band limits.")
                # if other.anType == 'L':
                #     data = []
                #     for idx, value in enumerate(self.data):
                #         d = 10*np.log10(10**(value/10) /
                #                         10**(other.data[idx]/10))
                #         data.append(d)
                #     anType = 'L'
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
        """
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
        h5group.attrs['comment'] = self.comment
        h5group['data'] = self.data
        return

    def plot(self, **kwargs):
        self.plot_bars(**kwargs)
        return

    def plot_bars(self, xlabel=None, ylabel=None, title=None, decimalSep=','):
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
        if xlabel is None:
            xlabel = 'Frequency bands [Hz]'
        if ylabel is None:
            ylabel = 'Modulus [{}]'
            ylabel = ylabel.format(self.unit)
        if title is None:
            title = '{} analysis'
            title = title.format(self.anName)


        
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
            minval += np.sign(minval)
            ax.bar(*zip(*enumerate(-minval + self.data)), width=0.75)
        else:
            ax.bar(fbar, self.data, width=0.75)
            minval = 0
        
        ylimInf = min(-minval + self.data) - 0.2
        ylimSup = max(-minval + self.data) + 0.2
        ylim = (ylimInf, ylimSup)
        ax.set_ylim(ylim)
        
        ax.grid(color='gray', linestyle='-.', linewidth=0.4)

        ax.set_xticks(fbar)
        xticks = self.bands
        ax.set_xticklabels(['{:n}'.format(tick) for tick in xticks],
                           rotation=45, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=20)

        yticks = np.linspace(*ylim, 11)
        ax.set_yticks(yticks.tolist())
        yticklabels = yticks + minval
        ax.set_yticklabels(['{:n}'.format(float('{0:.2f}'.format(tick)))
                            for tick in yticklabels.tolist()], fontsize=14)

        ax.set_ylabel(ylabel, fontsize=20)
            
        
        plt.title(title, fontsize=20)
        
        return
