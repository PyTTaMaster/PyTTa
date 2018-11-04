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
import setuptools

# with open("README.md","r") as fh:
# 	long_description = fh.read()

setuptools.setup(
	name = 'PyTTA',
	version = '0.0.0a1', # ALPHA version release
	author = ['Marcos Reis',
	    	'Matheus Lazarin',
		'João Vitor Paes'],
	author_email = ['marcos.reis@eac.ufsm.br',
			'matheus.lazarin@eac.ufsm.br',
			'joao.paes@eac.ufsm.br'],
	description = 'Signal processing tools for acoustics and vibrations in python.',
#	long_description = long_description,
#	long_description_content_type = 'text/markdown',
	url = 'http://github.com/pytta',
	packages = setuptools.find_packages(),
	zip_safe =  False,
	license = 'LGPL',
	install_requires = ['numpy',
			   'scipy',
			   'matplotlib',
			   'sounddevice',
			   'pyfilterbank'],
	classifiers=[
    		# How mature is this project? Common values are
		#   3 - Alpha
		#   4 - Beta
    		#   5 - Production/Stable
    		'Development Status :: 3 - Alpha',

	        # Indicate who your project is intended for
    		'Intended Audience :: Developers',
    		'Topic :: Software Development :: Build Tools',

    		# Pick your license as you wish (should match "license" above)
     		'License :: OSI Approved :: MIT License',

    		# Specify the Python versions you support here. In particular, ensure
    		# that you indicate whether you support Python 2, Python 3 or both.
    		'Programming Language :: Python :: 3.4',
    		'Programming Language :: Python :: 3.5',
    		'Programming Language :: Python :: 3.6',
	    	'Programming Language :: Python :: 3.7']

#    'package_data': {  # apenas para arquivos com outras extensões (.c, .h, .exe, etc)
#        'pytta': [
#            'sosfilt.c',
#            'sosfilt64.dll',
#            'sosfilt32.dll',
#            'sosfilt.so'
#        ]
#    }

)
