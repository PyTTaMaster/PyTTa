# -*- coding: utf-8 -*-
"""
PyTTa setup file
=================

Authors:
    João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
    Matheus Lazarin Alberto, matheus.lazarin@eac.ufsm.br

"""

from setuptools import setup
from glob import glob

with open("README.md", "r") as f:
    long_description = f.read()

settings = {
    'name': 'PyTTa',
    'version': '0.1.0',
    'description': 'Signal processing tools for acoustics and vibrations in ' +
        'python, development package.',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'url': 'http://github.com/PyTTAmaster/PyTTa',
    'author': 'João Vitor Paes, Matheus Lazarin, Marcos Reis',
    'author_email': 'pytta@eac.ufsm.br',
    'license': 'MIT',
    'install_requires': ['numpy', 'scipy', 'matplotlib',
        'sounddevice', 'soundfile', 'h5py', 'numba'],
    'packages': ['pytta', 'pytta.classes', 'pytta.apps', 'pytta.utils'],
    'data_files':  [('examples', glob('examples/*'))],
    # 'package_dir': {'classes': 'pytta'},
    'classifiers': [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", ],
    # 'package_data': {'pytta': ['examples/*.py', 'examples/RIS/*.mat']}
}

setup(**settings)
