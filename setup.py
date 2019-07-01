# -*- coding: utf-8 -*-
"""
PyTTa setup file
=================

@Autor:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

"""

import os
from setuptools import setup

settings = {
    'name': 'PyTTa',
    'version': '0.1.0b',
    'description': 'Signal processing tools for acoustics and vibrations in python.',
    'url': 'http://github.com/PyTTAmaster/PyTTa',
    'author': 'Marcos Reis, Matheus Lazarin, João Vitor Paes',
    'author_email': 'joao.paes@eac.ufsm.br',
    'license': 'MIT',
    'install_requires': ['numpy','scipy', 'matplotlib', 'sounddevice', 'soundfile'],
    'packages': ['pytta', 'pytta.classes'],
    'package_dir': {'classes': 'pytta'},
    'package_data': {'pytta': '..'+os.sep+'examples/*.py'}
}

setup(**settings)
