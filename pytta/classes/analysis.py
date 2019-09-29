# -*- coding: utf-8 -*-

from pytta.classes.filter import fractional_octave_frequencies as FOF
import numpy as np
import time

class Analysis(object):
    """
    """

    # Magic methods

    def __init__(self,
                anType,
                nthOct,
                bandMin,
                bandMax,
                data,
                comment='No comments.'):
        self.anType = anType
        self.nthOct = nthOct
        self.bandMin = bandMin
        self.bandMax = bandMax
        self.data = data
        self.comment = comment
        return

    def __str__(self):
        return ('1/{} octave band {} '.format(self.nthOct, self.anType) +
            'analysis from the {} [Hz] to the '.format(self.bandMin) +
            '{} [Hz] band.'.format(self.bandMax))

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'anType={self.anType!r}, '
                f'nthOct={self.nthOct!r}, '
                f'bandMin={self.bandMin!r}, '
                f'bandMax={self.bandMax!r}, '
                f'data={self.data!r}, '
                f'comment={self.comment!r})')

    def __add__(self, other):
        # check for min/max bands
        raise NotImplementedError

    def __sub__(self, other):
        # check for min/max bands
        raise NotImplementedError

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
    def anType(self, new):
        if type(new) is not str:
            raise TypeError("anType parameter makes reference to the module \
                             used to generate the analysis, e.g. 'room', \
                             'building', and must be a str value.")
        self._anType = new
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
                # TO DO: convertion calculation
                pass
        else:
            self._nthOct = new
        return


    @property
    def bandMin(self):
        """
        """
        return self._bandMin

    @bandMin.setter
    def bandMin(self, new):
        if type(new) is not int and type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._bandMin = new
        return

    @property
    def bandMax(self):
        """
        """
        return self._bandMax

    @bandMax.setter
    def bandMax(self, new):
        if type(new) is not int and type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._bandMax = new
        return

    @property
    def data(self):
        """
        """
        return self._data

    @data.setter
    def data(self, newData):
        bands = FOF(nthOct=self.nthOct,
                    minFreq=self.bandMin,
                    maxFreq=self.bandMax)
        if not isinstance(newData, list) and not isinstance(newData, np.ndarray):
            raise TypeError("'data' must be provided as a list or " +
                            "numpy ndarray.")
        elif len(newData) != len(bands):
            raise ValueError("Provided 'data' has different number of bands " +
                             "then the existant bands betwen " +
                             "{} and {} [Hz].".format(self.bandMin,
                                                      self.bandMax))
        
        # ...
        self._data = 0
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

    def plot_bars(self):
        return
