#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 18:34:15 2020

@author: mtslazarin
"""

import pytta

sin1 = pytta.generate.sin(freq=500)
sin2 = pytta.generate.sin(freq=400)
sin3 = pytta.generate.sin(freq=100)

sins = pytta.merge(sin1, sin2, sin3)

#%%
sins.play(channels=[2,3],mapping=[1,2])

#%%
sins.play(channels=[3],mapping=[1,2])

sins.play(channels=[3])

#%%
sins.play(channels=[1,3],mapping=[2,1])