"""
Tests for the creation of a SignalObj from inside a function.

SignalObj reads the name of the variable it is assigned to from the call
stack (classes/_instanceinfo.py). The lookup must not depend on what the
calling function holds in its first local variable. Each test pins one
case that used to raise.
"""
import os
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import pytta


def from_array(data):
    # first local variable: a NumPy array
    return pytta.SignalObj(data, 'time', 48000)


def before_first_local_is_set():
    # first local variable not assigned yet when SignalObj is created
    sig = pytta.SignalObj(np.ones(8), 'time', 48000)
    return sig


class TestSignalObjCreation(unittest.TestCase):

    def test_caller_holding_an_array(self):
        self.assertEqual(from_array(np.ones(8)).numSamples, 8)

    def test_caller_whose_first_local_is_not_set(self):
        self.assertEqual(before_first_local_is_set().numSamples, 8)


if __name__ == '__main__':
    unittest.main()
