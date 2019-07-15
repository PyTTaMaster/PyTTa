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
    'name': 'PyTTa-dev',
    'version': '0.1.0b4',
    'description': 'Signal processing tools for acoustics and vibrations in python, development package.',
    'url': 'http://github.com/PyTTAmaster/PyTTa',
    'author': 'Marcos Reis, Matheus Lazarin, João Vitor Paes',
    'author_email': 'joao.paes@eac.ufsm.br',
    'license': 'MIT',
    'install_requires': ['numpy','scipy', 'matplotlib', 'sounddevice', 'soundfile'],
    'packages': ['pytta', 'pytta.classes'],
    'package_dir': {'classes': 'pytta'},
    'package_data': {'pytta': ['examples/*.py', 'examples/RIS/*.mat']}
}

setup(**settings)
