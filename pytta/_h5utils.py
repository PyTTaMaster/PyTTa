# -*- coding: utf-8 -*-
"""
h5utilities:
-----------

    This submodule carries a set of internal functions for packing and
    unpacking h5py groups.

    Available functions:
    ---------------------

        >>> pytta.h5utilities.none_parser( attr )


    For further information, check the function specific documentation.
"""

import numpy as np

def attr_parser(attr):
    """
    Parser for all possible transformations for hdf5 data input and output.
    """
    attr = int_parser(attr)
    attr = float_parser(attr)
    attr = none_parser(attr)
    return attr


def int_parser(attr):
    if isinstance(attr, (np.int16,
                         np.int32,
                         np.int64,
                         np.int,
                         np.int0)):
        return int(attr)
    else:
        return attr

def float_parser(attr):
    if isinstance(attr, (np.float,
                        np.float16,
                        np.float32,
                        np.float64)):
        return float(attr)
    else:
        return attr


def none_parser(attr):
    if isinstance(attr, str):
        if attr == 'None':
            return None
        else:
            return attr
    if attr is None:
        return 'None'
    else:
        return attr


def list_w_int_parser(attr):
    if isinstance(attr, list):
        return str(attr)
    elif isinstance(attr, str):
        if attr[0] == '[' and attr[-1:] == ']':
            # removing '[' and ']'
            newlist = attr.replace('[', '').replace(']', '')
            # removing ','
            newlist = newlist.replace(',', '')
            # splitting elements
            newlist = newlist.split(' ')
            # constructing new list with ints
            return [int(x) for x in newlist]
    else:
        return attr
