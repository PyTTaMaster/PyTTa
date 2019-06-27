# -*- coding: utf-8 -*-


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

    def plot(self, style):
        pass


class ResultList(object):
    """
    """
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def add_property(self, propName, propVal):
        setattr(self, propName, propVal)
        return
