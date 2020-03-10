# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:29:33 2020

@author: Christof
"""


import numpy as np

depth = 15
width = 5
total = 2**17
mod = 0
summe = 0

a = np.ones(depth)

while summe < total:
    mod = mod + 1
    for i in range(depth):
        a[i] = width + mod * i

    summe = 2 * np.sum(a[:-1] * a[1:])

tempwidth = np.append(np.flip(a),a[1:])
tempdepth = 2 * depth -1
print('Depth', tempdepth)
print('Size of Tempwidth', tempwidth.shape[0])
print(a[:-1])
print(a[1:])
print('Summe: ', summe)
print('Modifier: ', mod)
