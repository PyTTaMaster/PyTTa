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
    'version': '0.1.1',
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
    # 'package_dir': {'classes': 'pytta'},
    # 'package_data': {'pytta': ['examples/*.py', 'examples/RIS/*.mat']}
    # 'data_files': [('examples', glob('examples/*'))],
    'classifiers': [
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", ],
    'python_requires': '>=3.6, <3.9',
}

setup(**settings)
