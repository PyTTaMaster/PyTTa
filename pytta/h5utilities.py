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


def none_parser(attr):
    if attr != 'None' and attr is not None:
        return attr
    elif attr == 'None':
        return None
    elif attr is None:
        return 'None'


def list_w_int_parser(attr):
    if isinstance(attr, list):
        return str(attr)
    if isinstance(attr, str):
        if attr[0] == '[' and attr[-1:] == ']':
            # removing '[' and ']'
            newlist = attr.replace('[', '').replace(']', '')
            # removing ','
            newlist = newlist.replace(',', '')
            # spliting elements
            newlist = newlist.split(' ')
            # constructing new list with ints
            return [int(x) for x in newlist]