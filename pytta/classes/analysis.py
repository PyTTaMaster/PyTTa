# -*- coding: utf-8 -*-

from pytta.classes.filter import fractional_octave_frequencies as FOF
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import time

# Analysis types and its units
anTypes = {'RT': 's',
           'C': 'dB',
           'D': '%',
           'mixed': '-'}

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
        # check for min/max bands
        raise NotImplementedError

    def __sub__(self, other):
        if other.range != self.range:
            raise ValueError("Can't subtract! Both Analysis have different " +
                             "band limits.")
        result = Analysis(anType='mixed', nthOct=self.nthOct,
                          minBand=self.minBand, maxBand=self.maxBand,
                          data=self.data-other.data)
        return result

    def __mul__(self, other):
        # check for min/max bands
        raise NotImplementedError

    def __truediv__(self, other):
        # check for min/max bands
        raise NotImplementedError

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
        self.unit = anTypes[newType]
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

    def _to_dict(self):
        return

    def pytta_save(self, dirname=time.ctime(time.time())):
        return

    def h5_save(self, h5group):
        """
        Saves itself inside a hdf5 group from an already openned file via
        pytta.save(...).
        """
        return

    def plot(self):
        self.plot_bars()
        return

    def plot_bars(self, xlabel=None, ylabel=None):
        """
        Analysis bar plotting method
        """
        if xlabel is None:
            xlabel = 'Frequency bands [Hz]'
        if ylabel is None:
            ylabel = 'Modulus [{}]'

        ylabel = ylabel.format(self.unit)

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_axes([0.08, 0.15, 0.75, 0.8], polar=False,
                          projection='rectilinear', xscale='linear')
        ax.set_snap(True)

        fbar = range(len(self.data))

        ax.bar(fbar, self.data, width=0.75)
        
        ax.grid(color='gray', linestyle='-.', linewidth=0.4)

        ax.set_xticks(fbar)
        xticks = self.bands
        ax.set_xticklabels(['{:n}'.format(tick) for tick in xticks],
                           rotation=45, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=20)
        

        ylimInf = min(self.data) - 0.2
        ylimSup = max(self.data) + 0.2
        ylim = (ylimInf, ylimSup)

        ax.set_ylim(ylim)
        yticks = np.linspace(*ylim, 11).tolist()
        ax.set_yticks(yticks)
        ax.set_yticklabels(['{:n}'.format(float('{0:.2f}'.format(tick)))
                            for tick in yticks], fontsize=14)

        ax.set_ylabel(ylabel, fontsize=20)
        
        return
