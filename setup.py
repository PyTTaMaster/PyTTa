# -*- coding: utf-8 -*-
"""
PyTTa setup file
=================

@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

"""

from setuptools import setup

settings = {
    'name': 'PyTTa',
    'version': '0.1.0b7',
    'description': 'Signal processing tools for acoustics and vibrations in ' +
        'python, development package.',
    'url': 'http://github.com/PyTTAmaster/PyTTa',
    'author': 'Marcos Reis, Matheus Lazarin, João Vitor Paes',
    'author_email': 'pytta@eac.ufsm.br',
    'license': 'MIT',
    'install_requires': ['numpy', 'scipy', 'matplotlib',
        'sounddevice', 'soundfile', 'h5py', 'numba'],
    'packages': ['pytta', 'pytta.classes', 'pytta.apps'],
    'package_dir': {'classes': 'pytta'},
    'package_data': {'pytta': ['examples/*.py', 'examples/RIS/*.mat']}
}

setup(**settings)
