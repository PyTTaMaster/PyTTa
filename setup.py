# -*- coding: utf-8 -*-
"""
PyTTa setup file
=================

@Autor:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

"""

#%%
from setuptools import setup

settings = {
    'name': 'PyTTa',
    'version': '0.1.0b1',
    'description': 'Signal processing tools for acoustics and vibrations in python.',
    'url': 'http://github.com/PyTTAmaster/PyTTa',
    'author': 'Marcos Reis, Matheus Lazarin, João Vitor Paes',
    'packages': ['pytta'],
    'zip_safe': False,
    'author_email': 'joao.paes@eac.ufsm.br',
    'license': 'LGPL',
    'requires': ['numpy','scipy','sounddevice','pyfilterbank'],
#    'package_data': {
#        'pytta': [
#            'sosfilt.c',
#            'sosfilt64.dll',
#            'sosfilt32.dll',
#            'sosfilt.so'
#        ]
#    }
}
setup(**settings)
