# -*- coding: utf-8 -*-

from pytta.classes.filter import fractional_octave_frequencies as FOF
import time

class Analysis(object):
    """
    """

    # Magic methods

    def __init__(self, anType=None, freqRange=None):
        self.freqRange = freqRange
        self.anType = anType
        return

    def __str__(self):
        strself = self.anType.title() + ' analysis from '\
                  + str(self.freqRange[0]) + ' [Hz] to '\
                  + str(self.freqRange[1]) + ' [Hz].'
        return strself

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        # check for min/max bands
        return

    def __sub__(self, other):
        # check for min/max bands
        return

    def __mul__(self, other):
        # check for min/max bands
        return

    def __truediv__(self, other):
        # check for min/max bands
        return

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
    def generator(self):
        """
        """
        return self._generator

    @generator.setter
    def generator(self, new):
        if type(new) is not str:
            raise TypeError('generator parameter makes reference to ' +
                            'the module used to generate the analysis, ' +
                            'e.g. \'room\', \'building\', and must be ' +
                            'a str value.')
        self._generator = new
        return

    @property
    def minBand(self):
        """
        """
        return self._freqRange[0]

    @minBand.setter
    def minBand(self, new):
        if type(new) is not int or type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._freqRange[0] = new
        return

    @property
    def maxBand(self):
        """
        """
        return self._freqRange[-1]

    @maxBand.setter
    def maxBand(self, new):
        if type(new) is not int or type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._freqRange[-1] = new
        return

    @property
    def bands(self):
        """
        """
        return self._freqRange

    @bands.setter
    def bands(self, new):
        if type(new) is not list:
            raise TypeError("Frequency range must be a list of int \
                            or a list of float values.")
        self._freqRange = new[:]
        return

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
