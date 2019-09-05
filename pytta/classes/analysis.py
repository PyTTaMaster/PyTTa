# -*- coding: utf-8 -*-


from .filter import fractional_octave_frequencies


class Result(object):
    """
    """
    def __init__(self, vals):
        self._values = vals
        pass

    def __repr__(self):
        return str(self._values)

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass


class ResultList(object):
    """
    """
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
    def minFreq(self):
        """
        """
        return self._freqRange[0]

    @minFreq.setter
    def minFreq(self, new):
        if type(new) is not int or type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._freqRange[0] = new
        return

    @property
    def maxFreq(self):
        """
        """
        return self._freqRange[-1]

    @maxFreq.setter
    def maxFreq(self, new):
        if type(new) is not int or type(new) is not float:
            raise TypeError("Frequency range values must \
                            be either int or float.")
        self._freqRange[-1] = new
        return

    @property
    def freqRange(self):
        """
        """
        return self._freqRange

    @freqRange.setter
    def freqRange(self, new):
        if type(new) is not list:
            raise TypeError("Frequency range must be a list of int \
                            or a list of float values.")
        self._freqRange = new[:]
        return

    @property
    def freqAxis(self):
        return self._freqAxis

    def get_x_axis(self):
        pass

    def plot(self, which, how='bars'):
        pass

    def add_attr(self, propName, propVal):
        """
        """
        if len(propVal) < len(self.freqAxis):
            raise ValueError("This list has " + str(len(self.freqAxis))
                             + " elements at x axis. New properties must have \
                             the same number of elements.")
        setattr(self, propName, propVal)
        return
