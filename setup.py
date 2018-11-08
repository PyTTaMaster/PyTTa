# -*- coding: utf-8 -*-
"""
                     PyTTa
    Object Oriented Python in Technical Acoustics

                    Setup
  
@Autores:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br


@Última modificação: 27/10/18

"""

#%%
from setuptools import setup

settings = {
    'name': 'PyTTa',
    'version': '0.0.0b',
    'description': 'Signal processing tools for acoustics and vibrations in python.',
    'url': 'http://github.com/pytta',
    'author': 'Marcos Reis, Matheus Lazarin, João Vitor Paes',
    'packages': ['pytta'],
    'zip_safe': False,
    'author_email': 'joao.paes@eac.ufsm.br',
    'license': 'LGPL',
    'install_requires': ['numpy','scipy','sounddevice','pyfilterbank'],
#    'package_data': {
#        'pyfilterbank': [
#            'sosfilt.c',
#            'sosfilt64.dll',
#            'sosfilt32.dll',
#            'sosfilt.so'
#        ]
#    }
}
setup(**settings)
